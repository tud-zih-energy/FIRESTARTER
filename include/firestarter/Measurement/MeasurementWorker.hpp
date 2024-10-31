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

#include "Metric/IPCEstimate.hpp"
#include "Metric/Perf.hpp"
#include "Metric/RAPL.hpp"
#include "MetricInterface.h"
#include "Summary.hpp"
#include "TimeValue.hpp"
#include "firestarter/WindowsCompat.hpp" // IWYU pragma: keep
#include <chrono>
#include <map>
#include <mutex>

void insertCallback(void* Cls, const char* MetricName, int64_t TimeSinceEpoch, double Value);

namespace firestarter::measurement {

class MeasurementWorker {
private:
  pthread_t WorkerThread{};
  pthread_t StdinThread{};

  std::vector<const MetricInterface*> Metrics = {&RaplMetric, &PerfIpcMetric, &PerfFreqMetric, &IpcEstimateMetric};

  std::mutex ValuesMutex;
  std::map<std::string, std::vector<TimeValue>> Values;

  static auto dataAcquisitionWorker(void* MeasurementWorker) -> void*;

  static auto stdinDataAcquisitionWorker(void* MeasurementWorker) -> void*;

  auto findMetricByName(std::string MetricName) -> const MetricInterface*;

  std::chrono::milliseconds UpdateInterval;

  std::chrono::high_resolution_clock::time_point StartTime;

  // some metric values have to be devided by this
  const uint64_t NumThreads;

  std::string AvailableMetricsString;

#ifndef FIRESTARTER_LINK_STATIC
  std::vector<void*> MetricDylibs;
#endif

  std::vector<std::string> StdinMetrics;

public:
  // creates the worker thread
  MeasurementWorker(std::chrono::milliseconds UpdateInterval, uint64_t NumThreads,
                    std::vector<std::string> const& MetricDylibsNames,
                    std::vector<std::string> const& StdinMetricsNames);

  // stops the worker threads
  ~MeasurementWorker();

  [[nodiscard]] auto availableMetrics() const -> std::string const& { return this->AvailableMetricsString; }

  auto stdinMetrics() -> std::vector<std::string> const& { return StdinMetrics; }

  // returns a list of metrics
  auto metricNames() -> std::vector<std::string>;

  // setup the selected metrics
  // returns a vector with the names of inialized metrics
  auto initMetrics(std::vector<std::string> const& MetricNames) -> std::vector<std::string>;

  // callback function for metrics
  void insertCallback(const char* MetricName, int64_t TimeSinceEpoch, double Value);

  // start the measurement
  void startMeasurement();

  // get the measurement values begining from measurement start until now.
  auto getValues(std::chrono::milliseconds StartDelta = std::chrono::milliseconds::zero(),
                 std::chrono::milliseconds StopDelta = std::chrono::milliseconds::zero())
      -> std::map<std::string, Summary>;
};

} // namespace firestarter::measurement
