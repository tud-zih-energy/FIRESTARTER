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

#include "firestarter/Measurement/Metric.hpp"
#include "firestarter/Measurement/Metric/IPCEstimate.hpp"
#include "firestarter/Measurement/Metric/Perf.hpp"
#include "firestarter/Measurement/Metric/RAPL.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <sstream>
#include <thread>

namespace firestarter::measurement {

/// This class handles the management of metrics, acquisition of metric data and provids summaries of a time range of
/// metric values.
class MeasurementWorker {
private:
  /// The thread that handles the values that are read from metrics
  std::thread WorkerThread;
  /// The thread that handles the metric values that are read from stdin
  std::thread StdinThread;

  /// The signal that terminated the threads.
  std::atomic<bool> StopExecution = false;

  /// The vector of metrics that are available. Currently the following metrics are builtin: sysfs-powercap-rapl,
  /// perf-ipc, perf-freq and ipc-estimate. Metric provided through shared libraries are added to this list.
  std::vector<std::shared_ptr<RootMetric>> Metrics = {
      RootMetric::fromCInterface(RaplMetric), RootMetric::fromCInterface(PerfIpcMetric),
      RootMetric::fromCInterface(PerfFreqMetric), RootMetric::fromCInterface(IpcEstimateMetric)};

  /// We poll the values of all the metrics after this number of milliseconds.
  std::chrono::milliseconds UpdateInterval;

  /// The start time of the measurement that should be summarized with the getValues function.
  std::chrono::high_resolution_clock::time_point StartTime;

  /// The number of thread FIRESTARTER runs with. This is required by some metrics
  const uint64_t NumThreads;

  /// The thread function handles the timed polling of the metric values and saves them to the Value datastructure.
  static void dataAcquisitionWorker(MeasurementWorker& This);

  /// The thread function that handles the acquisition of the metric values from stdin and saves them to the Value
  /// datastructure.
  static void stdinDataAcquisitionWorker(MeasurementWorker& This);

  /// Return the pointer to a metric from the Metrics vector that matches the supplied name.
  /// \arg MetricName The name of the metric
  /// \returns the pointer to the metric with the specified name or a nullptr
  auto findRootMetricByName(const MetricName& Metric) -> std::shared_ptr<RootMetric> {
    auto NameEqual = [&Metric](auto const& RootMetric) { return RootMetric->Name == Metric.rootMetricName(); };
    auto MetricReference = std::find_if(Metrics.begin(), Metrics.end(), NameEqual);
    if (MetricReference == Metrics.end()) {
      return {};
    }
    return *MetricReference;
  }

  /// Get all metrics matching a filter function defined on the std::shared_ptr<RootMetric>
  /// \arg FilterFunction The function that take a shared_ptr to the RootMetric and returns
  /// true if the metric names of this metric should be saved.
  /// \return The vector of filtered metric names.
  auto
  metrics(const std::function<bool(const std::shared_ptr<RootMetric>&)>& FilterFunction) -> std::vector<MetricName> {
    std::vector<MetricName> MetricNames;
    for (const auto& Metric : Metrics) {
      const auto Names = Metric->getMetricNames();

      if (FilterFunction(Metric)) {
        MetricNames.insert(MetricNames.end(), Names.cbegin(), Names.cend());
      }
    }
    return MetricNames;
  }

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

  /// Get the names of all available metrics
  auto availableMetrics() -> std::vector<MetricName> {
    return metrics(
        /*FilterFunction=*/[](const auto& RootMetric) { return RootMetric->Available; });
  }

  /// Get the names of all initialized metrics
  auto initializedMetrics() -> std::vector<MetricName> {
    return metrics(/*FilterFunction=*/[](const auto& RootMetric) { return RootMetric->Initialized; });
  }

  /// Get the names of all metrics
  auto metrics() -> std::vector<MetricName> {
    return metrics(/*FilterFunction=*/[](const auto& /*RootMetric*/) { return true; });
  }

  /// Get the formatting table of all metrics and if they are available
  [[nodiscard]] auto availableMetricsString() const -> std::string {
    std::stringstream Ss;
    unsigned MaxLength = 0;

    for (const auto& Metric : Metrics) {
      const auto Names = Metric->getMetricNames();

      for (const auto& Name : Names) {
        MaxLength = MaxLength < Name.toString().size() ? Name.toString().size() : MaxLength;
      }
    }

    const auto Padding = MaxLength > 6 ? MaxLength - 6 : 0;
    Ss << "  METRIC" << std::string(Padding + 1, ' ') << "| available\n";
    Ss << "  " << std::string(Padding + 7, '-') << "-----------\n";

    for (auto const& Metric : Metrics) {
      const auto Names = Metric->getMetricNames();

      for (const auto& Name : Names) {
        Ss << "  " << Name.toString() << std::string(Padding + 7 - Name.toString().size(), ' ') << "| ";
        Ss << (Metric->Available ? "yes" : "no") << "\n";
      }
    }

    return Ss.str();
  }

  /// Initialize the metrics with the provided names.
  /// \arg MetricNames The metrics to initialize
  /// \returns The vector of metrics that were successfully initialized.
  void initMetrics(std::vector<MetricName> const& MetricNames);

  /// Set the StartTime to the current timestep
  void startMeasurement();

  /// Get the measurement values begining from measurement start (set with startMeasurement) until the measurement stop
  /// (now).
  /// \arg StartDelta The time to skip from the measurement start
  /// \arg StopDelta The time to skip from the measurement stop
  /// \returns The map from all metrics to their respective summaries.
  auto getValues(std::chrono::milliseconds StartDelta = std::chrono::milliseconds::zero(),
                 std::chrono::milliseconds StopDelta = std::chrono::milliseconds::zero()) -> MetricSummaries;
};

} // namespace firestarter::measurement
