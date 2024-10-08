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

static const std::string RaplPath = "/sys/class/powercap";

static std::string errorString;

struct ReaderDef {
  char* Path;
  long long int LastReading;
  long long int Overflow;
  long long int Max;
};

struct ReaderDefFree {
  void operator()(struct ReaderDef* Def) {
    if (Def != nullptr) {
      if (((void*)Def->Path) != nullptr) {
        free((void*)Def->Path);
      }
      free((void*)Def);
    }
  }
};

static std::vector<std::shared_ptr<struct ReaderDef>> Readers = {};

static auto fini() -> int32_t {
  Readers.clear();

  return EXIT_SUCCESS;
}

static auto init() -> int32_t {
  errorString = "";

  DIR* RaplDir = opendir(RaplPath.c_str());
  if (RaplDir == nullptr) {
    errorString = "Could not open " + RaplPath;
    return EXIT_FAILURE;
  }

  // we try to find psys first
  // then package + dram
  // and finally package only.

  // contains an empty path if it is not found
  std::string PsysPath;

  // a vector of all paths to package and dram
  std::vector<std::string> Paths = {};

  struct dirent* Dir = nullptr;
  while ((Dir = readdir(RaplDir)) != nullptr) {
    std::stringstream Path;
    std::stringstream NamePath;
    Path << RaplPath << "/" << Dir->d_name;
    NamePath << Path.str() << "/name";

    std::ifstream NameStream(NamePath.str());
    if (!NameStream.good()) {
      // an error opening the file occured
      continue;
    }

    std::string Name;
    std::getline(NameStream, Name);

    if (Name == "psys") {
      // found psys
      PsysPath = Path.str();
    } else if (0 == Name.rfind("package", 0) || Name == "dram") {
      // find all package and dram
      Paths.push_back(Path.str());
    }
  }
  closedir(RaplDir);

  // make psys the only value if available
  if (!PsysPath.empty()) {
    Paths.clear();
    Paths.push_back(PsysPath);
  }

  // paths now contains all interesting nodes

  if (Paths.empty()) {
    errorString = "No valid entries in " + RaplPath;
    return EXIT_FAILURE;
  }

  for (auto const& Path : Paths) {
    std::stringstream EnergyUjPath;
    EnergyUjPath << Path << "/energy_uj";
    std::ifstream EnergyReadingStream(EnergyUjPath.str());
    if (!EnergyReadingStream.good()) {
      errorString = "Could not read energy_uj";
      break;
    }

    std::stringstream MaxEnergyUjRangePath;
    MaxEnergyUjRangePath << Path << "/max_energy_range_uj";
    std::ifstream MaxEnergyReadingStream(MaxEnergyUjRangePath.str());
    if (!MaxEnergyReadingStream.good()) {
      errorString = "Could not read max_energy_range_uj";
      break;
    }

    uint64_t Reading = 0;
    uint64_t Max = 0;
    std::string Buffer;
    int Read = 0;

    std::getline(EnergyReadingStream, Buffer);
    Read = std::sscanf(Buffer.c_str(), "%lu", &Reading);

    if (Read == 0) {
      std::stringstream Ss;
      Ss << "Contents in file " << EnergyUjPath.str() << " do not conform to mask (uint64_t)";
      errorString = Ss.str();
      break;
    }

    std::getline(MaxEnergyReadingStream, Buffer);
    Read = std::sscanf(Buffer.c_str(), "%lu", &Max);

    if (Read == 0) {
      std::stringstream Ss;
      Ss << "Contents in file " << MaxEnergyUjRangePath.str() << " do not conform to mask (uint64_t)";
      errorString = Ss.str();
      break;
    }

    std::shared_ptr<struct ReaderDef> Def(reinterpret_cast<struct ReaderDef*>(malloc(sizeof(struct ReaderDef))),
                                          ReaderDefFree());
    const auto* PathName = Path.c_str();
    size_t Size = (strlen(PathName) + 1) * sizeof(char);
    void* Name = malloc(Size);
    memcpy(Name, PathName, Size);
    Def->Path = (char*)Name;
    Def->Max = Max;
    Def->LastReading = Reading;
    Def->Overflow = 0;

    Readers.push_back(Def);
  }

  if (!errorString.empty()) {
    fini();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static auto getReading(double* Value) -> int32_t {
  double FinalReading = 0.0;

  for (auto& Def : Readers) {
    long long int Reading = 0;
    std::string Buffer;

    std::stringstream EnergyUjPath;
    EnergyUjPath << Def->Path << "/energy_uj";
    std::ifstream EnergyReadingStream(EnergyUjPath.str());
    std::getline(EnergyReadingStream, Buffer);
    std::sscanf(Buffer.c_str(), "%llu", &Reading);

    if (Reading < Def->LastReading) {
      Def->Overflow += 1;
    }

    Def->LastReading = Reading;

    FinalReading += 1.0E-6 * (double)((Def->Overflow * Def->Max) + Def->LastReading);
  }

  if (Value != nullptr) {
    *Value = FinalReading;
  }

  return EXIT_SUCCESS;
}

static auto getError() -> const char* {
  const char* ErrorCString = errorString.c_str();
  return ErrorCString;
}

// this function will be called periodically to make sure we do not miss an
// overflow of the counter
static void callback() { getReading(nullptr); }
}

const MetricInterface RaplMetric = {
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
    .GetReading = getReading,
    .GetError = getError,
    .RegisterInsertCallback = nullptr,
};
