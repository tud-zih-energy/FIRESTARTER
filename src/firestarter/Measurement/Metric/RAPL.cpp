/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
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
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

extern "C" {
#include <firestarter/Measurement/Metric/RAPL.h>
#include <firestarter/Measurement/MetricInterface.h>

#include <dirent.h>

#define RAPL_PATH "/sys/class/powercap"

static std::string errorString = "";

struct reader_def {
  char* path;
  long long int last_reading;
  long long int overflow;
  long long int max;
};

struct reader_def_free {
  void operator()(struct reader_def* def) {
    if (def != nullptr) {
      if (((void*)def->path) != nullptr) {
        free((void*)def->path);
      }
      free((void*)def);
    }
  }
};

static std::vector<std::shared_ptr<struct reader_def>> readers = {};

static int32_t fini(void) {
  readers.clear();

  return EXIT_SUCCESS;
}

static int32_t init(void) {
  errorString = "";

  DIR* raplDir = opendir(RAPL_PATH);
  if (raplDir == NULL) {
    errorString = "Could not open " RAPL_PATH;
    return EXIT_FAILURE;
  }

  // we try to find psys first
  // then package + dram
  // and finally package only.

  // contains an empty path if it is not found
  std::string psysPath = "";

  // a vector of all paths to package and dram
  std::vector<std::string> paths = {};

  struct dirent* dir;
  while ((dir = readdir(raplDir)) != NULL) {
    std::stringstream path;
    std::stringstream namePath;
    path << RAPL_PATH << "/" << dir->d_name;
    namePath << path.str() << "/name";

    std::ifstream nameStream(namePath.str());
    if (!nameStream.good()) {
      // an error opening the file occured
      continue;
    }

    std::string name;
    std::getline(nameStream, name);

    if (name == "psys") {
      // found psys
      psysPath = path.str();
    } else if (0 == name.rfind("package", 0) || name == "dram") {
      // find all package and dram
      paths.push_back(path.str());
    }
  }
  closedir(raplDir);

  // make psys the only value if available
  if (!psysPath.empty()) {
    paths.clear();
    paths.push_back(psysPath);
  }

  // paths now contains all interesting nodes

  if (paths.size() == 0) {
    errorString = "No valid entries in " RAPL_PATH;
    return EXIT_FAILURE;
  }

  for (auto const& path : paths) {
    std::stringstream energyUjPath;
    energyUjPath << path << "/energy_uj";
    std::ifstream energyReadingStream(energyUjPath.str());
    if (!energyReadingStream.good()) {
      errorString = "Could not read energy_uj";
      break;
    }

    std::stringstream maxEnergyUjRangePath;
    maxEnergyUjRangePath << path << "/max_energy_range_uj";
    std::ifstream maxEnergyReadingStream(maxEnergyUjRangePath.str());
    if (!maxEnergyReadingStream.good()) {
      errorString = "Could not read max_energy_range_uj";
      break;
    }

    uint64_t reading;
    uint64_t max;
    std::string buffer;
    int read;

    std::getline(energyReadingStream, buffer);
    read = std::sscanf(buffer.c_str(), "%lu", &reading);

    if (read == 0) {
      std::stringstream ss;
      ss << "Contents in file " << energyUjPath.str() << " do not conform to mask (uint64_t)";
      errorString = ss.str();
      break;
    }

    std::getline(maxEnergyReadingStream, buffer);
    read = std::sscanf(buffer.c_str(), "%lu", &max);

    if (read == 0) {
      std::stringstream ss;
      ss << "Contents in file " << maxEnergyUjRangePath.str() << " do not conform to mask (uint64_t)";
      errorString = ss.str();
      break;
    }

    std::shared_ptr<struct reader_def> def(reinterpret_cast<struct reader_def*>(malloc(sizeof(struct reader_def))),
                                           reader_def_free());
    auto pathName = path.c_str();
    size_t size = (strlen(pathName) + 1) * sizeof(char);
    void* name = malloc(size);
    memcpy(name, pathName, size);
    def->path = (char*)name;
    def->max = max;
    def->last_reading = reading;
    def->overflow = 0;

    readers.push_back(def);
  }

  if (errorString.size() != 0) {
    fini();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static int32_t get_reading(double* value) {
  double finalReading = 0.0;

  for (auto& def : readers) {
    long long int reading;
    std::string buffer;

    std::stringstream energyUjPath;
    energyUjPath << def->path << "/energy_uj";
    std::ifstream energyReadingStream(energyUjPath.str());
    std::getline(energyReadingStream, buffer);
    std::sscanf(buffer.c_str(), "%llu", &reading);

    if (reading < def->last_reading) {
      def->overflow += 1;
    }

    def->last_reading = reading;

    finalReading += 1.0E-6 * (double)(def->overflow * def->max + def->last_reading);
  }

  if (value != nullptr) {
    *value = finalReading;
  }

  return EXIT_SUCCESS;
}

static const char* get_error(void) {
  const char* errorCString = errorString.c_str();
  return errorCString;
}

// this function will be called periodically to make sure we do not miss an
// overflow of the counter
static void callback() { get_reading(nullptr); }
}

MetricInterface RaplMetric = {
    .Name = "sysfs-powercap-rapl",
    .Type = {.Absolute = 0,
             .Accumalative = 1,
             .DivideByThreadCount = 0,
             .InsertCallback = 0,
             .IgnoreStartStopDelta = 0,
             .Reserved = 0},
    .Unit = "J",
    .CallbackTime = 30000000,
    .Callback = callback,
    .Init = init,
    .Fini = fini,
    .GetReading = get_reading,
    .GetError = get_error,
    .RegisterInsertCallback = nullptr,
};
