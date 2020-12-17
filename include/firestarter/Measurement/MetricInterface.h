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

#include <stdint.h>

// clang-format off
typedef struct {
  // Either set absolute or accumalative to specify the type of values from the
  // metric.
  uint32_t absolute : 1,
           accumalative : 1,
           // Set to divide metric values by thread count.
           divide_by_thread_count : 1,
           // Set to insert time-value pairs via callback function passed by
           // register_insert_callback.
           insert_callback : 1,
           __reserved : 28;
} metric_type_t;
// clang-format on

// Define `metric_interface_t metric` inside your shared library to be able to
// load it during runtime.
typedef struct {
  // the name of the metric
  const char *name;

  // metric type with bitfield from metric_type_t
  metric_type_t type;

  // the unit of the metric
  const char *unit;

  uint64_t callback_time;

  // This function will be called every `callback_time` usecs. Disable by
  // setting `callback_time` to 0.
  void (*callback)(void);

  // init the metric.
  // returns EXIT_SUCCESS on success.
  int32_t (*init)(void);

  // deinit the metric.
  // returns EXIT_SUCCESS on success.
  int32_t (*fini)(void);

  // Get a reading of the metric
  // Return EXIT_SUCCESS if we got a new value.
  // Set this function pointer to NULL if METRIC_INSERT_CALLBACK is specified.
  int32_t (*get_reading)(double *value);

  // Get error in case return code not being EXIT_SUCCESS
  const char *(*get_error)(void);

  // If METRIC_INSERT_CALLBACK is set in the type, this function will be passed
  // a callback and the first argument for the callback.
  // Further arguments of callback are the metric name, an unix timestamp (time
  // since epoch) and a metric value.
  int32_t (*register_insert_callback)(void (*)(void *, const char *, int64_t,
                                               double),
                                      void *);

} metric_interface_t;
