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

#pragma once

#include "firestarter/Measurement/MetricInterface.h"

#include <string>

/// The wrapper for the C interface to the IpcEstimateMetric metric.
struct IpcEstimateMetric {
private:
  IpcEstimateMetric() = default;

  /// The error string of this metric
  std::string ErrorString;

  /// The saved callback to push the metric value
  void (*Callback)(void*, uint64_t, int64_t, double){};
  /// The saved first argument for the callback
  void* CallbackArg{};

public:
  IpcEstimateMetric(IpcEstimateMetric const&) = delete;
  void operator=(IpcEstimateMetric const&) = delete;

  /// Get the instance of this metric
  static auto instance() -> IpcEstimateMetric& {
    static IpcEstimateMetric Instance;
    return Instance;
  }

  /// Deinit the metric.
  /// \returns EXIT_SUCCESS on success.
  static auto fini() -> int32_t;

  /// Init the metric.
  /// \returns EXIT_SUCCESS on success.
  static auto init() -> int32_t;

  /// Get error in case return code not being EXIT_SUCCESS.
  /// \returns The error string.
  static auto getError() -> const char*;

  /// The first argument is the function pointer to the callback. The first argument to this function pointer needs to
  /// be filled with the second argument to this function.
  /// The supplied function pointer needs to be called with either zero in case the metric value is provided or the
  /// index starting with one of the submetric, an unix timestamp (time since epoch) for the third and a metric value
  /// for the forth argument. This allows the metric to provide values in a pushing way in contrast to the pulling way
  /// of the GetReading function.
  static auto registerInsertCallback(void (*C)(void*, uint64_t, int64_t, double), void* Arg) -> int32_t;

  /// Push a value with the current timestamp.
  /// \arg Value The metric value to push.
  static void insertValue(double Value);

  /// This metric provdies the ipc estimated based on the estimated number of instructions and the runtime of the high
  /// load loop. The metric value is dependent on the frequency of the processor. It serves as an estimation of the IPC
  /// times the processor frequency.
  inline static MetricInterface Metric{
      /*Name=*/"ipc-estimate",
      /*Type=*/
      {/*Absolute=*/1, /*Accumalative=*/0, /*DivideByThreadCount=*/0, /*InsertCallback=*/1, /*IgnoreStartStopDelta=*/1,
       /*Reserved=*/0},
      /*Unit=*/"IPC",
      /*CallbackTime=*/0,
      /*Callback=*/nullptr,
      /*Init=*/init,
      /*Fini=*/fini,
      /*GetSubmetricNames=*/
      nullptr,
      /*GetReading=*/nullptr,
      /*GetError=*/getError,
      /*RegisterInsertCallback=*/registerInsertCallback,
  };
};