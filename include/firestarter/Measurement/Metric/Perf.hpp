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

#include "../MetricInterface.h"
#include <string>

struct PerfMetricData {
  inline static const char* PerfEventParanoidFile = "/proc/sys/kernel/perf_event_paranoid";

  struct ReadFormat {
    uint64_t Nr;
    struct {
      uint64_t Value;
      uint64_t Id;
    } Values[2];
  };

  inline static std::string ErrorString;
  inline static int CpuCyclesFd = -1;
  inline static int InstructionsFd = -1;
  inline static uint64_t CpuCyclesId;
  inline static uint64_t InstructionsId;
  inline static bool InitDone = false;
  inline static int32_t InitValue;
  inline static struct ReadFormat Last;
  static auto fini() -> int32_t;
  static auto init() -> int32_t;
  static auto valueFromId(struct ReadFormat* Values, uint64_t Id) -> uint64_t;
  static auto getReading(double* IpcValue, double* FreqValue) -> int32_t;
  static auto getReadingIpc(double* Value) -> int32_t;
  static auto getReadingFreq(double* Value) -> int32_t;
  static auto getError() -> const char*;
};

static constexpr const MetricInterface PerfIpcMetric{
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

static constexpr const MetricInterface PerfFreqMetric{
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