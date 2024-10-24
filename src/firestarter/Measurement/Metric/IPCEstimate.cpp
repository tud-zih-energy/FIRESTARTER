/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2021 TU Dresden, Center for Information Services and High
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

#include <chrono>
#include <cstdlib>
#include <firestarter/Measurement/Metric/IPCEstimate.hpp>
#include <string>

auto IpcEstimateMetricData::fini() -> int32_t {
  Callback = nullptr;
  CallbackArg = nullptr;

  return EXIT_SUCCESS;
}

auto IpcEstimateMetricData::init() -> int32_t {
  ErrorString = "";

  return EXIT_SUCCESS;
}

auto IpcEstimateMetricData::getError() -> const char* {
  const char* ErrorCString = ErrorString.c_str();
  return ErrorCString;
}

auto IpcEstimateMetricData::registerInsertCallback(void (*C)(void*, const char*, int64_t, double), void* Arg)
    -> int32_t {
  Callback = C;
  CallbackArg = Arg;
  return EXIT_SUCCESS;
}

void ipcEstimateMetricInsert(double Value) {
  if (IpcEstimateMetricData::Callback == nullptr || IpcEstimateMetricData::CallbackArg == nullptr) {
    return;
  }

  int64_t T =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();

  IpcEstimateMetricData::Callback(IpcEstimateMetricData::CallbackArg, "ipc-estimate", T, Value);
}