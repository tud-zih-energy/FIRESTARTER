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
#include <string>

extern "C" {
#include <firestarter/Measurement/Metric/IPCEstimate.h>
#include <firestarter/Measurement/MetricInterface.h>
}

static std::string ErrorString;

static void (*Callback)(void*, const char*, int64_t, double) = nullptr;
static void* CallbackArg = nullptr;

static auto fini() -> int32_t {
  Callback = nullptr;
  CallbackArg = nullptr;

  return EXIT_SUCCESS;
}

static auto init() -> int32_t {
  ErrorString = "";

  return EXIT_SUCCESS;
}

static auto getError() -> const char* {
  const char* ErrorCString = ErrorString.c_str();
  return ErrorCString;
}

static auto registerInsertCallback(void (*C)(void*, const char*, int64_t, double), void* Arg) -> int32_t {
  Callback = C;
  CallbackArg = Arg;
  return EXIT_SUCCESS;
}

void ipcEstimateMetricInsert(double Value) {
  if (Callback == nullptr || CallbackArg == nullptr) {
    return;
  }

  int64_t T =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();

  Callback(CallbackArg, "ipc-estimate", T, Value);
}

const MetricInterface IpcEstimateMetric = {
    .Name = "ipc-estimate",
    .Type = {.Absolute = 1,
             .Accumalative = 0,
             .DivideByThreadCount = 0,
             .InsertCallback = 1,
             .IgnoreStartStopDelta = 1,
             .Reserved = 0},
    .Unit = "IPC",
    .CallbackTime = 0,
    .Callback = nullptr,
    .Init = init,
    .Fini = fini,
    .GetReading = nullptr,
    .GetError = getError,
    .RegisterInsertCallback = registerInsertCallback,
};
