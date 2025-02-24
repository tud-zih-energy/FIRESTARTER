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

#include "firestarter/Measurement/Metric/RAPL.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <dirent.h>
}

auto RaplMetricData::fini() -> int32_t {
  instance().Readers.clear();
  instance().AccumulateReaders.clear();
  instance().SubmetricNames = {nullptr};

  return EXIT_SUCCESS;
}

auto RaplMetricData::init() -> int32_t {
  auto& Instance = instance();

  Instance.ErrorString = "";

  DIR* RaplDir = opendir(RaplPath);
  if (RaplDir == nullptr) {
    Instance.ErrorString = "Could not open " + std::string(RaplPath);
    return EXIT_FAILURE;
  }

  // a vector of all paths to package and dram
  std::vector<std::string> Paths = {};

  struct dirent* Dir = nullptr;

  // As long as the DIR object (named RaplDir here) is not shared between threads this call is thread-safe:
  // https://www.gnu.org/software/libc/manual/html_node/Reading_002fClosing-Directory.html
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while ((Dir = readdir(RaplDir)) != nullptr) {
    std::stringstream Path;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
    Path << RaplPath << "/" << Dir->d_name;

    try {
      auto Def = std::make_shared<ReaderDef>(/*Path=*/Path.str());
      Instance.Readers.emplace_back(std::move(Def));
    } catch (std::invalid_argument const& E) {
      // This path does not contain a metric
      continue;
    } catch (std::runtime_error const& E) {
      Instance.ErrorString = E.what();
      break;
    }
  }
  closedir(RaplDir);

  // we try to find psys first
  // then package + dram
  // and finally package only.

  // nullptr if not found
  std::shared_ptr<ReaderDef> PsysReader;

  for (const auto& Reader : Instance.Readers) {
    const auto& Name = Reader->name();
    if (Name.find("psys") != std::string::npos) {
      PsysReader = Reader;
    } else if (Name.find("dram") != std::string::npos || Name.find("package") != std::string::npos) {
      Instance.AccumulateReaders.emplace_back(Reader);
    }
  }

  if (PsysReader) {
    Instance.AccumulateReaders = {PsysReader};
  }

  // check that we have readers.
  if (Instance.AccumulateReaders.empty()) {
    Instance.ErrorString = "No valid entries in " + std::string(RaplPath);
    return EXIT_FAILURE;
  }

  if (!Instance.ErrorString.empty()) {
    fini();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

auto RaplMetricData::getReading(double* Value) -> int32_t {
  double FinalReading = 0.0;

  // Update all readers
  for (auto& Reader : instance().Readers) {
    Reader->read();
  }

  // Read the value
  for (const auto& Reader : instance().AccumulateReaders) {
    FinalReading += Reader->lastReading();
  }

  if (Value != nullptr) {
    *Value = FinalReading;
  }

  return EXIT_SUCCESS;
}

auto RaplMetricData::getError() -> const char* {
  const char* ErrorCString = instance().ErrorString.c_str();
  return ErrorCString;
}

// this function will be called periodically to make sure we do not miss an
// overflow of the counter
void RaplMetricData::callback() { getReading(nullptr); }