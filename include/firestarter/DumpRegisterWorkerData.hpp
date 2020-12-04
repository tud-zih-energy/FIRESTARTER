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

#pragma once

#include <firestarter/LoadWorkerData.hpp>

#include <chrono>

namespace firestarter {

/* DO NOT CHANGE! the asm load-loop tests if it should dump the current register
 * content */
enum DumpVariable : unsigned long long { Start = 0, Wait = 1 };

#define REGISTER_MAX_NUM 32

struct DumpRegisterStruct {
  // REGISTER_MAX_NUM cachelines
  volatile double registerValues[REGISTER_MAX_NUM * 8];
  // pad to use a whole cacheline
  volatile unsigned long long padding[7];
  volatile DumpVariable dumpVar;
};

#undef REGISTER_MAX_NUM

} // namespace firestarter

#ifdef DEBUG_FEATURES

// current mingw gcc version 8 has a bug, which causes this include to abort the
// compilation
#include <filesystem>

namespace firestarter {

class DumpRegisterWorkerData {
public:
  DumpRegisterWorkerData(LoadWorkerData *loadWorkerData,
                         std::chrono::seconds dumpTimeDelta,
                         std::string dumpFilePath)
      : loadWorkerData(loadWorkerData), dumpTimeDelta(dumpTimeDelta) {
    // if dumpFilePath is empty, use current working directory
    if (dumpFilePath.empty()) {
      this->dumpFilePath = std::filesystem::current_path();
    } else {
      this->dumpFilePath = std::filesystem::path(dumpFilePath);
    }
  };
  ~DumpRegisterWorkerData() {}

  LoadWorkerData *const loadWorkerData;
  const std::chrono::seconds dumpTimeDelta;
  std::filesystem::path dumpFilePath;
};

} // namespace firestarter

#endif
