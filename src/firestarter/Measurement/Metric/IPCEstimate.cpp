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

static std::string errorString = "";

static void (*callback)(void*, const char*, int64_t, double) = nullptr;
static void* callback_arg = nullptr;

static int32_t fini(void) {
  callback = nullptr;
  callback_arg = nullptr;

  return EXIT_SUCCESS;
}

static int32_t init(void) {
  errorString = "";

  return EXIT_SUCCESS;
}

static const char* get_error(void) {
  const char* errorCString = errorString.c_str();
  return errorCString;
}

static int32_t register_insert_callback(void (*c)(void*, const char*, int64_t, double), void* arg) {
  callback = c;
  callback_arg = arg;
  return EXIT_SUCCESS;
}

void ipcEstimateMetricInsert(double Value) {
  if (callback == nullptr || callback_arg == nullptr) {
    return;
  }

  int64_t t =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();

  callback(callback_arg, "ipc-estimate", t, Value);
}

MetricInterface IpcEstimateMetric = {
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
    .GetError = get_error,
    .RegisterInsertCallback = register_insert_callback,
};
