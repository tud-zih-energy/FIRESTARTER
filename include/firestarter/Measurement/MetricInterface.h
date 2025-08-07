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

/// This file provides a C style interface to write metrics for FIRESTARTER and provide them as shared libraries.

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdint.h>

// NOLINTBEGIN(modernize-use-using)

/// Describe the type of the metric and how values need to be accumulated. Per default metrics are of pulling type where
/// FIRESTARTER will pull the values through the GetReading function.

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ROOT_METRIC_INDEX 0

typedef struct {
  uint32_t
      /// Set this to 1 if the metric values provided are absolute.
      Absolute : 1,
      /// Set this to 1 if the metric values provided are accumulative.
      Accumalative : 1,
      /// Set this to 1 if the metric value needs to be divided by the number of threads.
      DivideByThreadCount : 1,
      /// Set this to 1 if the metric will provide time-value data in a pushing way trough the RegisterInsertCallback
      /// function.
      InsertCallback : 1,
      /// Set this to 1 if the accumulation of the metric should ignore the start/stop delta which are specified by the
      /// user of FIRESTARTER.
      IgnoreStartStopDelta : 1,
      /// Reserved space to fill 32 bits
      Reserved : 27;
} MetricType;

/// Define `MetricInterface Metric` inside your shared library to be able to load it during runtime.
typedef struct {
  /// The name of the metric
  const char* Name;

  /// Describes what the value of the metrics represents and how it needs to be accumulated.
  MetricType Type;

  /// The unit of the metric
  const char* Unit;

  /// The time in usecs after which the callback should be called again. Set to 0 to disable.
  uint64_t CallbackTime;

  /// This function will be called every `CallbackTime` usecs. Disable by setting `CallbackTime` to 0.
  void (*Callback)();

  /// init the metric.
  /// \returns EXIT_SUCCESS on success.
  int32_t (*Init)();

  /// deinit the metric.
  /// \returns EXIT_SUCCESS on success.
  int32_t (*Fini)();

  /// Get a vector of submetric names. This is required to know the name of a submetric that is just described via an
  /// index throughout this metric interface.
  /// This function ptr may be NULL in case the metric does not support submetrics.
  /// \returns The NULL terminated array of submetric names (char *)
  const char** (*GetSubmetricNames)();

  /// Get a reading of the metric. Set this function pointer to null if MetricType::InsertCallback is specified in the
  /// Type.
  /// \arg Values The memory array to which double values are saved. The index zero contains the root metric. The values
  /// one and up are used to select the specific submetric.
  /// \arg NumElems The number of elements in the double array.
  /// \returns EXIT_SUCCESS if we got a new value.
  int32_t (*GetReading)(double* Value, uint64_t NumElems);

  /// Get error in case return code not being EXIT_SUCCESS.
  /// \returns The error string.
  const char* (*GetError)();

  /// If MetricType::InsertCallback is specified in the Type this function will be used to pass the metric a callback
  /// and the first argument to this callback.
  /// The first argument is the function pointer to the callback. The first argument to this function pointer needs to
  /// be filled with the second argument to this function.
  /// The supplied function pointer needs to be called with either zero in case the metric value is provided or the
  /// index starting with one of the submetric, an unix timestamp (time since epoch) for the third and a metric value
  /// for the forth argument. This allows the metric to provide values in a pushing way in contrast to the pulling way
  /// of the GetReading function.
  int32_t (*RegisterInsertCallback)(void (*)(void*, uint64_t, int64_t, double), void*);

} MetricInterface;
// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
};
#endif