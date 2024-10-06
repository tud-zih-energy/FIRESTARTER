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

#include "LoadWorkerData.hpp"
#include "Logging/Log.hpp"
#include <chrono>
#include <utility>

namespace firestarter {

class DumpRegisterWorkerData {
public:
  DumpRegisterWorkerData() = delete;

  DumpRegisterWorkerData(std::shared_ptr<LoadWorkerData> LoadWorkerDataPtr, std::chrono::seconds DumpTimeDelta,
                         const std::string& DumpFilePath)
      : LoadWorkerDataPtr(std::move(LoadWorkerDataPtr))
      , DumpTimeDelta(DumpTimeDelta) {
    if (DumpFilePath.empty()) {
      char Cwd[PATH_MAX];
      if (getcwd(Cwd, sizeof(Cwd)) != nullptr) {
        this->DumpFilePath = Cwd;
      } else {
        log::error() << "getcwd() failed. Set --dump-registers-outpath to /tmp";
        this->DumpFilePath = "/tmp";
      }
    } else {
      this->DumpFilePath = DumpFilePath;
    }
  }

  ~DumpRegisterWorkerData() = default;

  std::shared_ptr<LoadWorkerData> LoadWorkerDataPtr;
  const std::chrono::seconds DumpTimeDelta;
  std::string DumpFilePath;
};

} // namespace firestarter