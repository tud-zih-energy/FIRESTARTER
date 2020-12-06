/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
 * Performance Computing
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/\>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

extern "C" {
#include <firestarter/Measurement/Metric/RAPL.h>
#include <firestarter/Measurement/MetricInterface.h>

#define RAPL_PATH "/sys/class/powercap"

static const char *unit = std::string("J").c_str();
static unsigned long long callback_time = 30000000;
static std::string errorString = "";

struct reader_def {
  const char *path;
  long long int last_reading;
  long long int overflow;
  long long int max;
};

static std::vector<struct reader_def *> readers = {};

static int fini(void) {
  for (auto &def : readers) {
    free(def);
  }

  readers.clear();

  return EXIT_SUCCESS;
}

static int init(void) {
  errorString = "";

  if (!fs::exists(RAPL_PATH)) {
    errorString = "Could not open " RAPL_PATH;
    return EXIT_FAILURE;
  }

  // we try to find psys first
  // then package + dram
  // and finally package only.

  // contains an empty path if it is not found
  fs::path psysPath = fs::path();

  // a vector of all paths to package and dram
  std::vector<fs::path> paths = {};

  auto iterator = fs::directory_iterator(RAPL_PATH);

  for (auto const &p : iterator) {
    auto path = p.path();
    auto namePath = path / "name";

    // namePath does not exists for identity symlink intel-rapl
    if (!fs::exists(namePath)) {
      continue;
    }

    std::ifstream nameStream(namePath);
    if (!nameStream.good()) {
      // an error opening the file occured
      continue;
    }

    std::string name;
    std::getline(nameStream, name);

    if (name == "psys") {
      // found psys
      psysPath = path;
    } else if (0 == name.rfind("package", 0) || name == "dram") {
      // find all package and dram
      paths.push_back(path);
    }
  }

  // make psys the only value if available
  if (fs::path() != psysPath) {
    paths.clear();
    paths.push_back(psysPath);
  }

  // paths now contains all interesting nodes

  if (paths.size() == 0) {
    errorString = "No valid entries in " RAPL_PATH;
    return EXIT_FAILURE;
  }

  for (auto const &path : paths) {
    std::ifstream energyReadingStream(path / "energy_uj");
    if (!energyReadingStream.good()) {
      errorString = "Could not read energy_uj";
      break;
    }

    std::ifstream maxEnergyReadingStream(path / "max_energy_range_uj");
    if (!maxEnergyReadingStream.good()) {
      errorString = "Could not read max_energy_range_uj";
      break;
    }

    unsigned long long reading;
    unsigned long long max;
    std::string buffer;
    int read;

    std::getline(energyReadingStream, buffer);
    read = std::sscanf(buffer.c_str(), "%llu", &reading);

    if (read == 0) {
      std::stringstream ss;
      ss << "Contents in file " << path / "energy_uj"
         << " do not conform to mask (unsigned long long)";
      errorString = ss.str();
      break;
    }

    std::getline(maxEnergyReadingStream, buffer);
    max = std::sscanf(buffer.c_str(), "%llu", &max);

    if (max == 0) {
      std::stringstream ss;
      ss << "Contents in file " << path / "max_energy_range_uj"
         << " do not conform to mask (unsigned long long)";
      errorString = ss.str();
      break;
    }

    struct reader_def *def = reinterpret_cast<struct reader_def *>(
        malloc(sizeof(struct reader_def)));
    auto pathName = path.c_str();
    size_t size = (strlen(pathName) + 1) * sizeof(char);
    void *name = malloc(size);
    memcpy(name, pathName, size);
    def->path = reinterpret_cast<const char *>(name);
    def->max = max;
    def->last_reading = reading;
    def->overflow = 0;

    readers.push_back(std::ref(def));
  }

  if (errorString.size() != 0) {
    fini();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static int get_reading(double *value) {
  double finalReading = 0.0;

  for (auto &def : readers) {
    long long int reading;
    std::string buffer;

    std::ifstream energyReadingStream(fs::path(def->path) / "energy_uj");
    std::getline(energyReadingStream, buffer);
    std::sscanf(buffer.c_str(), "%llu", &reading);

    if (reading < def->last_reading) {
      def->overflow += 1;
    }

    def->last_reading = reading;

    finalReading +=
        1.0E-6 * (double)(def->overflow * def->max + def->last_reading);
  }

  if (value != nullptr) {
    *value = finalReading;
  }

  return EXIT_SUCCESS;
}

static const char *get_error(void) {
  const char *errorCString = errorString.c_str();
  return errorCString;
}

// this function will be called periodically to make sure we do not miss an
// overflow of the counter
static void callback(void) { get_reading(nullptr); }
}

metric_interface_t rapl_metric = {.name = "sysfs-powercap-rapl",
                                  .type = METRIC_ACCUMALATIVE,
                                  .unit = unit,
                                  .callback_time = callback_time,
                                  .callback = callback,
                                  .init = init,
                                  .fini = fini,
                                  .get_reading = get_reading,
                                  .get_error = get_error};
