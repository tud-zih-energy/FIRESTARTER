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
#include "WindowsCompat.hpp" // IWYU pragma: keep
#include <chrono>
#include <utility>

namespace firestarter {

/// This class holds the data that is required for the worker thread that dumps the register contents to a file.
class DumpRegisterWorkerData {
public:
  DumpRegisterWorkerData() = delete;

  /// Initialize the DumpRegisterWorkerData.
  /// \arg LoadWorkerDataPtr The shared pointer to the data of the thread were registers should be dummped. We need it
  /// to access the memory to which the registers are dumped as well as getting the size and count of registers.
  /// \arg DumpTimeDelta Every this number of seconds the register content will be dumped.
  /// \arg DumpFilePath The folder that is used to dump registers to. If the string is empty the current directory will
  /// be choosen. If it cannot be determined /tmp is used. In this directory a file called hamming_distance.csv will be
  /// created.
  DumpRegisterWorkerData(std::shared_ptr<LoadWorkerData> LoadWorkerDataPtr, std::chrono::seconds DumpTimeDelta,
                         const std::string& DumpFilePath)
      : LoadWorkerDataPtr(std::move(LoadWorkerDataPtr))
      , DumpTimeDelta(DumpTimeDelta) {
    if (DumpFilePath.empty()) {
      char* Pwd = get_current_dir_name();
      if (Pwd) {
        this->DumpFilePath = Pwd;
      } else {
        log::error() << "getcwd() failed. Set --dump-registers-outpath to /tmp";
        this->DumpFilePath = "/tmp";
      }
    } else {
      this->DumpFilePath = DumpFilePath;
    }
  }

  ~DumpRegisterWorkerData() = default;

  /// The shared pointer to the data of the thread were registers should be dummped. We need it to access the memory to
  /// which the registers are dumped as well as getting the size and count of registers.
  std::shared_ptr<LoadWorkerData> LoadWorkerDataPtr;
  /// Every this number of seconds the register content will be dumped.
  const std::chrono::seconds DumpTimeDelta;
  /// The folder in which the hamming_distance.csv file will be created.
  std::string DumpFilePath;
};

} // namespace firestarter