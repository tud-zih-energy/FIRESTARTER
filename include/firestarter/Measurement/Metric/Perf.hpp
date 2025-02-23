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

#include "firestarter/Measurement/MetricInterface.h"

#include <array>
#include <string>

/// The wrapper for the C interface to the PerfIpcMetric and PerfFreqMetric metric.
class PerfMetricData {
private:
  PerfMetricData() = default;

  static const constexpr char* PerfEventParanoidFile = "/proc/sys/kernel/perf_event_paranoid";

  /// The datastructure that is read from the file descriptor provided by the perf_event_open syscall.
  struct ReadFormat {
    struct ValueAndId {
      uint64_t Value;
      uint64_t Id;
    };

    uint64_t Nr;
    std::array<ValueAndId, 2> Values;
  };

  /// The error string of this metric
  std::string ErrorString;

  /// The file descriptor of the perf_event_open syscall for the PERF_COUNT_HW_CPU_CYCLES event. This file descriptor
  /// handles as a group for the other file descriptor.
  int CpuCyclesFd = -1;
  /// The file descriptor of the perf_event_open syscall for the PERF_COUNT_HW_INSTRUCTIONS event.
  int InstructionsFd = -1;
  /// The PERF_EVENT_IOC_ID for the cpu cycles file descriptor.
  uint64_t CpuCyclesId{};
  /// The PERF_EVENT_IOC_ID for the cpu instruction file descriptor.
  uint64_t InstructionsId{};

  /// The flag that stop init from being executed multiple times.
  bool InitDone = false;
  /// The value that is returned if the init function called multiple times.
  int32_t InitValue{};

  /// Save the last read metric for the perf-ipc metric. This value will be updated when the perf-ipc metric is read.
  struct ReadFormat Last {};

  /// Get a reading of the perf-freq and perf-ipc metric. Pointers can be nullptr.
  /// \arg IpcValue The pointer to which the value for ipc metric value will be saved.
  /// \arg FreqValue The pointer to which the value for freq metric value will be saved.
  /// \returns EXIT_SUCCESS if we got a new value.
  static auto getReading(double* IpcValue, double* FreqValue) -> int32_t;

public:
  PerfMetricData(PerfMetricData const&) = delete;
  void operator=(PerfMetricData const&) = delete;

  /// Get the instance of this metric
  static auto instance() -> PerfMetricData& {
    static PerfMetricData Instance;
    return Instance;
  }

  /// Deinit the metric.
  /// \returns EXIT_SUCCESS on success.
  static auto fini() -> int32_t;

  /// Init the metric.
  /// \returns EXIT_SUCCESS on success.
  static auto init() -> int32_t;

  /// Read the from a specific PERF_EVENT_IOC_ID out of the ReadFormat datastructure.
  /// \arg Reader The ReadFormat datastructure from which the value will be extracter
  /// \arg Id The PERF_EVENT_IOC_ID of the metric that should be read.
  static auto valueFromId(struct ReadFormat* Reader, uint64_t Id) -> uint64_t;

  /// Get a reading of the perf-ipc metric.
  /// \arg Value The pointer to which the value will be saved.
  /// \returns EXIT_SUCCESS if we got a new value.
  static auto getReadingIpc(double* Value) -> int32_t;

  /// Get a reading of the perf-freq metric.
  /// \arg Value The pointer to which the value will be saved.
  /// \returns EXIT_SUCCESS if we got a new value.
  static auto getReadingFreq(double* Value) -> int32_t;

  /// Get error in case return code not being EXIT_SUCCESS.
  /// \returns The error string.
  static auto getError() -> const char*;
};

/// This metric provides IPC measurement of the programm and all associated threads.
inline static MetricInterface PerfIpcMetric{
    /*Name=*/"perf-ipc",
    /*Type=*/
    {/*Absolute=*/1, /*Accumalative=*/0, /*DivideByThreadCount=*/0, /*InsertCallback=*/0, /*IgnoreStartStopDelta=*/0,
     /*Reserved=*/0},
    /*Unit=*/"IPC",
    /*CallbackTime=*/0,
    /*Callback=*/nullptr,
    /*Init=*/PerfMetricData::init,
    /*Fini=*/PerfMetricData::fini,
    /*GetReading=*/PerfMetricData::getReadingIpc,
    /*GetError=*/PerfMetricData::getError,
    /*RegisterInsertCallback=*/nullptr,
};

/// This metric provides frequency measurement on the CPUs used to execute the program on.
inline static MetricInterface PerfFreqMetric{
    /*Name=*/"perf-freq",
    /*Type=*/
    {/*Absolute=*/0, /*Accumalative=*/1, /*DivideByThreadCount=*/1, /*InsertCallback=*/0, /*IgnoreStartStopDelta=*/0,
     /*Reserved=*/0},
    /*Unit=*/"GHz",
    /*CallbackTime=*/0,
    /*Callback=*/nullptr,
    /*Init=*/PerfMetricData::init,
    /*Fini=*/PerfMetricData::fini,
    /*GetReading=*/PerfMetricData::getReadingFreq,
    /*GetError=*/PerfMetricData::getError,
    /*RegisterInsertCallback=*/nullptr,
};