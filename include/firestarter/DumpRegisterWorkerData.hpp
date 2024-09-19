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

#pragma once

#include <chrono>
#include <firestarter/DumpRegisterStruct.hpp>
#include <firestarter/LoadWorkerData.hpp>

#ifdef FIRESTARTER_DEBUG_FEATURES

namespace firestarter {

class DumpRegisterWorkerData {
public:
  DumpRegisterWorkerData(std::shared_ptr<LoadWorkerData> loadWorkerData, std::chrono::seconds dumpTimeDelta,
                         std::string dumpFilePath)
      : loadWorkerData(loadWorkerData)
      , dumpTimeDelta(dumpTimeDelta) {
    if (dumpFilePath.empty()) {
      char cwd[PATH_MAX];
      if (getcwd(cwd, sizeof(cwd)) != NULL) {
        this->dumpFilePath = cwd;
      } else {
        log::error() << "getcwd() failed. Set --dump-registers-outpath to /tmp";
        this->dumpFilePath = "/tmp";
      }
    } else {
      this->dumpFilePath = dumpFilePath;
    }
  }

  ~DumpRegisterWorkerData() {}

  std::shared_ptr<LoadWorkerData> loadWorkerData;
  const std::chrono::seconds dumpTimeDelta;
  std::string dumpFilePath;
};

} // namespace firestarter

#endif
