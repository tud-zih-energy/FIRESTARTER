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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// NOLINTBEGIN(modernize-use-using)
typedef struct {
  uint32_t
      // metric value is absolute
      Absolute : 1,
      // metric value accumulates
      Accumalative : 1,
      // Set to divide metric values by thread count.
      DivideByThreadCount : 1,
      // Set to insert time-value pairs via callback function passed by
      // register_insert_callback.
      InsertCallback : 1,
      // ignore the start and stop delta set by the user
      IgnoreStartStopDelta : 1,
      // Reserved space to round up to 32 bits
      Reserved : 27;
} MetricType;

// Define `metric_interface_t metric` inside your shared library to be able to
// load it during runtime.
typedef struct {
  // the name of the metric
  const char* Name;

  // metric type with bitfield from metric_type_t
  MetricType Type;

  // the unit of the metric
  const char* Unit;

  uint64_t CallbackTime;

  // This function will be called every `callback_time` usecs. Disable by
  // setting `callback_time` to 0.
  void (*Callback)();

  // init the metric.
  // returns EXIT_SUCCESS on success.
  int32_t (*Init)();

  // deinit the metric.
  // returns EXIT_SUCCESS on success.
  int32_t (*Fini)();

  // Get a reading of the metric
  // Return EXIT_SUCCESS if we got a new value.
  // Set this function pointer to NULL if METRIC_INSERT_CALLBACK is specified.
  int32_t (*GetReading)(double* Value);

  // Get error in case return code not being EXIT_SUCCESS
  const char* (*GetError)();

  // If METRIC_INSERT_CALLBACK is set in the type, this function will be passed
  // a callback and the first argument for the callback.
  // Further arguments of callback are the metric name, an unix timestamp (time
  // since epoch) and a metric value.
  int32_t (*RegisterInsertCallback)(void (*)(void*, const char*, int64_t, double), void*);

} MetricInterface;
// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
};
#endif