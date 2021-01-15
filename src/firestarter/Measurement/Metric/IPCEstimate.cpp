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

static void (*callback)(void *, const char *, int64_t, double) = nullptr;
static void *callback_arg = nullptr;

static int32_t fini(void) {
  callback = nullptr;
  callback_arg = nullptr;

  return EXIT_SUCCESS;
}

static int32_t init(void) {
  errorString = "";

  return EXIT_SUCCESS;
}

static const char *get_error(void) {
  const char *errorCString = errorString.c_str();
  return errorCString;
}

static int32_t register_insert_callback(void (*c)(void *, const char *, int64_t,
                                                  double),
                                        void *arg) {
  callback = c;
  callback_arg = arg;
  return EXIT_SUCCESS;
}

void ipc_estimate_metric_insert(double value) {
  if (callback == nullptr || callback_arg == nullptr) {
    return;
  }

  int64_t t = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::high_resolution_clock::now().time_since_epoch())
                  .count();

  callback(callback_arg, "ipc-estimate", t, value);
}

metric_interface_t ipc_estimate_metric = {
    .name = "ipc-estimate",
    .type = {.absolute = 1,
             .accumalative = 0,
             .divide_by_thread_count = 0,
             .insert_callback = 1,
             .ignore_start_stop_delta = 1,
             .__reserved = 0},
    .unit = "IPC",
    .callback_time = 0,
    .callback = nullptr,
    .init = init,
    .fini = fini,
    .get_reading = nullptr,
    .get_error = get_error,
    .register_insert_callback = register_insert_callback,
};
