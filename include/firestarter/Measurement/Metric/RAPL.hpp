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

class RaplMetricData {
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

private:
  static constexpr const char* RaplPath = "/sys/class/powercap";

  std::string ErrorString;

  std::vector<std::unique_ptr<ReaderDef>> Readers;

  RaplMetricData() = default;

public:
  RaplMetricData(RaplMetricData const&) = delete;
  void operator=(RaplMetricData const&) = delete;

  static auto instance() -> RaplMetricData& {
    static RaplMetricData Instance;
    return Instance;
  }

  /// Deinit the metric.
  /// \returns EXIT_SUCCESS on success.
  static auto fini() -> int32_t;

  /// Init the metric.
  /// \returns EXIT_SUCCESS on success.
  static auto init() -> int32_t;

  /// Get a reading of the sysfs-powercap-rapl metric.
  /// \arg Value The pointer to which the value will be saved.
  /// \returns EXIT_SUCCESS if we got a new value.
  static auto getReading(double* Value) -> int32_t;

  /// Get error in case return code not being EXIT_SUCCESS.
  /// \returns The error string.
  static auto getError() -> const char*;

  /// This function should be called every 30s. It will make shure that we do not miss an overflow of a counter and
  /// therefore get a wrong reading.
  static void callback();
};

/// This metric provides power measurements through the RAPL interface. Either psys measurement is choosen or if this is
/// not available the sum of packages and drams.
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