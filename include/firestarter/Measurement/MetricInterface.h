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

typedef enum {
  METRIC_ABSOLUTE = 1 << 0,
  METRIC_ACCUMALATIVE = 1 << 1,
  METRIC_DIVIDE_BY_THREAD_COUNT = 1 << 2,
} metric_type_t;

typedef struct {
  const char *name;

  // type with bitfield from metric_type_t
  uint32_t type;

  const char *unit;

  // get the time in microseconds, the callback function has to be called.
  // this is usefull if we have a counter and it will overrun periodically.
  // if the callbackTime is zero, then no callback will happen
  uint64_t callback_time;

  // this function will get called periodically
  void (*callback)(void);

  // init the metric.
  // returns EXIT_SUCCESS on success.
  // after calling this function, the unit and callbackTime must be initialized
  // and are ought not to be changed.
  int32_t (*init)(void);

  // deinit the metric.
  int32_t (*fini)(void);

  // get a reading of the metric
  // return EXIT_SUCCESS on if we got a new value.
  // else return EXIT_FAILURE
  int32_t (*get_reading)(double *value);

  // get the error in case of EXIT_FAILURE
  const char *(*get_error)(void);

} metric_interface_t;
