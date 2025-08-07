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

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <dirent.h>
}

auto RaplMetric::fini() -> int32_t {
  instance().Readers.clear();
  instance().AccumulateReaders.clear();
  instance().SubmetricNames = {nullptr};

  return EXIT_SUCCESS;
}

auto RaplMetric::init() -> int32_t {
  auto& Instance = instance();

  Instance.ErrorString = "";

  DIR* RaplDir = opendir(RaplPath);
  if (RaplDir == nullptr) {
    Instance.ErrorString = "Could not open " + std::string(RaplPath);
    return EXIT_FAILURE;
  }

  struct dirent* Dir = nullptr;

  // As long as the DIR object (named RaplDir here) is not shared between threads this call is thread-safe:
  // https://www.gnu.org/software/libc/manual/html_node/Reading_002fClosing-Directory.html
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while ((Dir = readdir(RaplDir)) != nullptr) {
    std::stringstream Path;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
    Path << RaplPath << "/" << Dir->d_name;

    std::shared_ptr<ReaderDef> Def;

    try {
      Def = std::make_shared<ReaderDef>(/*Path=*/Path.str());
    } catch (std::invalid_argument const& E) {
      // This path does not contain a metric
      continue;
    } catch (std::runtime_error const& E) {
      Instance.ErrorString = E.what();
      break;
    }

    Instance.Readers.emplace_back(Def);

    // Replace the nullptr with the new element and apend the nullptr again
    *Instance.SubmetricNames.rbegin() = Def->name().c_str();
    Instance.SubmetricNames.emplace_back(nullptr);
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

auto RaplMetric::getReading(double* Value, uint64_t NumElems) -> int32_t {

  // Update all readers
  auto& Readers = instance().Readers;

  for (auto& Reader : Readers) {
    Reader->read();
  }

  if (Value != nullptr) {
    assert(NumElems == 1 + Readers.size() &&
           "The number of elems is smaller than the number of reader plus the root metric.");

    // Read the value
    double FinalReading = 0.0;
    for (const auto& Reader : instance().AccumulateReaders) {
      FinalReading += Reader->lastReading();
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    *Value++ = FinalReading;
    for (auto& Reader : Readers) {
      *Value++ = Reader->lastReading();
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  return EXIT_SUCCESS;
}

auto RaplMetric::getError() -> const char* {
  const char* ErrorCString = instance().ErrorString.c_str();
  return ErrorCString;
}

// this function will be called periodically to make sure we do not miss an
// overflow of the counter
void RaplMetric::callback() { getReading(nullptr, /*NumElems=*/0); }