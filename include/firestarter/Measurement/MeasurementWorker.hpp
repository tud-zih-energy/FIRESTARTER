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

#include "firestarter/Measurement/Metric/IPCEstimate.hpp"
#include "firestarter/Measurement/Metric/Perf.hpp"
#include "firestarter/Measurement/Metric/RAPL.hpp"
#include "firestarter/Measurement/MetricInterface.h"
#include "firestarter/Measurement/Summary.hpp"
#include "firestarter/Measurement/TimeValue.hpp"
#include "firestarter/WindowsCompat.hpp" // IWYU pragma: keep

#include <chrono>
#include <map>
#include <mutex>

void insertCallback(void* Cls, const char* MetricName, int64_t TimeSinceEpoch, double Value);

namespace firestarter::measurement {

/// This class handles the management of metrics, acquisition of metric data and provids summaries of a time range of
/// metric values.
class MeasurementWorker {
private:
  /// The thread that handles the values that are read from metrics
  pthread_t WorkerThread{};
  /// The thread that handles the metric values that are read from stdin
  pthread_t StdinThread{};

  /// The vector of metrics that are available. Currently the following metrics are builtin: sysfs-powercap-rapl,
  /// perf-ipc, perf-freq and ipc-estimate. Metric provided through shared libraries are added to this list.
  std::vector<const MetricInterface*> Metrics = {&RaplMetric, &PerfIpcMetric, &PerfFreqMetric, &IpcEstimateMetric};

  /// Mutex to access the Values map.
  std::mutex ValuesMutex;
  /// Map from metric name to the vector of timevalues of this metric.
  std::map<std::string, std::vector<TimeValue>> Values;

  /// The thread function handles the timed polling of the metric values and saves them to the Value datastructure.
  static auto dataAcquisitionWorker(void* MeasurementWorker) -> void*;

  /// The thread function that handles the acquisition of the metric values from stdin and saves them to the Value
  /// datastructure.
  static auto stdinDataAcquisitionWorker(void* MeasurementWorker) -> void*;

  /// Return the pointer to a metric from the Metrics vector that matches the supplied name.
  /// \arg MetricName The name of the metric
  /// \returns the pointer to the metric with the specified name or a nullptr
  auto findMetricByName(std::string MetricName) -> const MetricInterface*;

  /// We poll the values of all the metrics after this number of milliseconds.
  std::chrono::milliseconds UpdateInterval;

  /// The start time of the measurement that should be summarized with the getValues function.
  std::chrono::high_resolution_clock::time_point StartTime;

  /// The number of thread FIRESTARTER runs with. This is required by some metrics
  const uint64_t NumThreads;

  std::string AvailableMetricsString;

#ifndef FIRESTARTER_LINK_STATIC
  /// The pointer to the metrics that are used for dynamic libraries. We need to save them seperately here to call
  /// dlclose later.
  std::vector<void*> MetricDylibs;
#endif

  /// The name of the metrics that are supplied from stdin.
  std::vector<std::string> StdinMetrics;

public:
  /// Initilize the measurement worker. It will spawn the threads for the polling of metic values.
  /// \arg UpdateInterval The polling time for metric updates.
  /// \arg NumThreads The number of thread FIRESTARTER is running with.
  /// \arg MetricDylibsNames The vector of files to which are passed to dlopen for using additional metrics from shared
  /// libraries.
  /// \arg StdinMetricsNames The vector of metric names that should be read in from stdin
  MeasurementWorker(std::chrono::milliseconds UpdateInterval, uint64_t NumThreads,
                    std::vector<std::string> const& MetricDylibsNames,
                    std::vector<std::string> const& StdinMetricsNames);

  /// Stops the worker threads
  ~MeasurementWorker();

  /// Get the formatting table of all metrics and if they are available
  [[nodiscard]] auto availableMetrics() const -> std::string const& { return this->AvailableMetricsString; }

  /// The vector of all metrics that are read from stdin
  auto stdinMetrics() -> std::vector<std::string> const& { return StdinMetrics; }

  /// Get the name of the metrics. This includes all metrics, builins, from dynamic libraries and metrics from stdin.
  auto metricNames() -> std::vector<std::string>;

  /// Initialize the metrics with the provided names.
  /// \arg MetricNames The metrics to initialize
  /// \returns The vector of metrics that were successfully initialized.
  auto initMetrics(std::vector<std::string> const& MetricNames) -> std::vector<std::string>;

  /// This function insert a time value pair for a specific metric. This function will be provided to metrics to allow
  /// them to push time value pairs.
  /// \arg MetricName The name of the metric for which values are inserted
  /// \arg TimeSinceEpoch The time since epoch of the time value pair
  /// \arg Value The value of the time value pair
  void insertCallback(const char* MetricName, int64_t TimeSinceEpoch, double Value);

  /// Set the StartTime to the current timestep
  void startMeasurement();

  /// Get the measurement values begining from measurement start (set with startMeasurement) until the measurement stop
  /// (now).
  /// \arg StartDelta The time to skip from the measurement start
  /// \arg StopDelta The time to skip from the measurement stop
  /// \returns The map from all metrics to their respective summaries.
  auto
  getValues(std::chrono::milliseconds StartDelta = std::chrono::milliseconds::zero(),
            std::chrono::milliseconds StopDelta = std::chrono::milliseconds::zero()) -> std::map<std::string, Summary>;
};

} // namespace firestarter::measurement
