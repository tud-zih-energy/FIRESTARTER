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

#include "../MetricInterface.h"
#include <string>

struct IpcEstimateMetricData {
  inline static std::string ErrorString;
  inline static void (*Callback)(void*, const char*, int64_t, double);
  inline static void* CallbackArg;
  static auto fini() -> int32_t;
  static auto init() -> int32_t;
  static auto getError() -> const char*;
  static auto registerInsertCallback(void (*C)(void*, const char*, int64_t, double), void* Arg) -> int32_t;
};

static constexpr const MetricInterface IpcEstimateMetric{
    /*Name=*/"ipc-estimate",
    /*Type=*/
    {/*Absolute=*/1, /*Accumalative=*/0, /*DivideByThreadCount=*/0, /*InsertCallback=*/1, /*IgnoreStartStopDelta=*/1,
     /*Reserved=*/0},
    /*Unit=*/"IPC",
    /*CallbackTime=*/0,
    /*Callback=*/nullptr,
    /*Init=*/IpcEstimateMetricData::init,
    /*Fini=*/IpcEstimateMetricData::fini,
    /*GetReading=*/nullptr,
    /*GetError=*/IpcEstimateMetricData::getError,
    /*RegisterInsertCallback=*/IpcEstimateMetricData::registerInsertCallback,
};

void ipcEstimateMetricInsert(double Value);