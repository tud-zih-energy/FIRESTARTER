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
#include <memory>
#include <string>
#include <vector>

struct RaplMetricData {
  inline static const char* const RaplPath = "/sys/class/powercap";

  inline static std::string ErrorString;

  struct ReaderDef {
    ReaderDef() = delete;

    ReaderDef(std::string Path, int64_t LastReading, int64_t Overflow, int64_t Max)
        : Path(std::move(Path))
        , LastReading(LastReading)
        , Overflow(Overflow)
        , Max(Max){};

    std::string Path;
    int64_t LastReading;
    int64_t Overflow;
    int64_t Max;
  };

  inline static std::vector<std::unique_ptr<ReaderDef>> Readers;

  static auto fini() -> int32_t;
  static auto init() -> int32_t;

  static auto getReading(double* Value) -> int32_t;

  static auto getError() -> const char*;

  static void callback();
};

static constexpr const MetricInterface RaplMetric{
    /*Name=*/"sysfs-powercap-rapl",
    /*Type=*/
    {/*Absolute=*/0, /*Accumalative=*/1, /*DivideByThreadCount=*/0, /*InsertCallback=*/0, /*IgnoreStartStopDelta=*/0,
     /*Reserved=*/0},
    /*Unit=*/"J",
    /*CallbackTime=*/30000000,
    /*Callback=*/RaplMetricData::callback,
    /*Init=*/RaplMetricData::init,
    /*Fini=*/RaplMetricData::fini,
    /*GetReading=*/RaplMetricData::getReading,
    /*GetError=*/RaplMetricData::getError,
    /*RegisterInsertCallback=*/nullptr,
};