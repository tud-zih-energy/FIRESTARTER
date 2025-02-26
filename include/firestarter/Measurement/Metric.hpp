/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2025 TU Dresden, Center for Information Services and High
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

#include "firestarter/Config/MetricName.hpp"
#include "firestarter/Measurement/MetricInterface.h"
#include "firestarter/Measurement/Summary.hpp"
#include "firestarter/Measurement/TimeValue.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace firestarter::measurement {

using MetricSummaries = std::map<MetricName, Summary>;

struct RootMetric;

/// This class handels the state around a metric. Its name, the contained time value pairs and a mutex to guard them.
struct Metric {
  /// The name of the metric
  std::string Name;
  /// The data collected from the metrics.
  std::vector<TimeValue> Values;
  /// Mutex to access the Values
  std::mutex ValuesMutex;

  Metric() = delete;

  explicit Metric(std::string Name)
      : Name(std::move(Name)) {}
};

/// This class handels the state around a leaf metric. Its name, the contained time value pairs and a mutex to guard
/// them.
struct LeafMetric : public Metric {
  /// The reference to the root metric that contains this leaf metric.
  const RootMetric& RootRef;

  LeafMetric() = delete;

  explicit LeafMetric(std::string Name, const RootMetric& RootRef)
      : Metric(std::move(Name))
      , RootRef(RootRef) {}

  /// Get the name of this leaf metric.
  auto metricName() -> MetricName;
};

/// This class handels the state around a root metric. Its name, the contained time value pairs, a mutex to guard them,
/// if it is initialized and available and the pointer to the interface to interact with it. Root metrics include
/// addition leaf-/sub- metrics.
struct RootMetric : public Metric {
  /// The pointer to the metric
  MetricInterface* MetricPtr = nullptr;
  /// The names of the submetrics.
  std::vector<std::unique_ptr<LeafMetric>> Submetrics;
  /// Is the metric from a dynamic library
  bool Dylib;
  /// Is the metric from stdin input?
  bool Stdin;
  /// Is the metric initialized
  bool Initialized = false;
  /// Is the metric available
  bool Available = false;

  RootMetric() = delete;

  /// \arg Name The name of the metric
  /// \arg Available Is the metric available
  RootMetric(std::string Name, bool Dylib, bool Stdin, bool Initialized)
      : Metric(std::move(Name))
      , Dylib(Dylib)
      , Stdin(Stdin)
      , Initialized(Initialized) {}

  ~RootMetric();

  /// Popuplate the RootMetric object from the C-style MetricInterface.
  /// \arg Metric the refernce to the C-style MetricInterface
  static auto fromCInterface(MetricInterface& Metric) -> std::shared_ptr<RootMetric>;

#if not(defined(FIRESTARTER_LINK_STATIC)) && defined(linux)
  /// Popuplate the RootMetric object from the C-style MetricInterface provided via a dynamic library.
  /// \arg DylibPath The dynamic library name
  static auto fromDylib(const std::string& DylibPath) -> std::shared_ptr<RootMetric>;
#endif

  /// Popuplate the RootMetric object from the C-style MetricInterface.
  /// \arg Metric the refernce to the C-style MetricInterface
  static auto fromStdin(const std::string& MetricName) -> std::shared_ptr<RootMetric>;

  /// Initialize the metric. This will set the state to inititialized if successful and insert the optional callback
  /// into the metric.
  auto initialize() -> bool;

  /// Get the callback function and callback interval that has to be executed in a timed fashion for the metric.
  /// \returns an empty optional if the metric does not required timed callbacks. otherwise it returns a tuple of the
  /// callback function and callback interval. The callback function should be called after the callback interval has
  /// expired.
  auto getTimedCallback() -> std::optional<std::tuple<std::function<void()>, std::chrono::microseconds>>;

  /// Get the optional callback that has to be executed to insert values into the metric storage.
  /// \return optionally returns the function that has to be executed to save the current metric value into the
  /// stoarage.
  auto getInsertCallback() -> std::optional<std::function<void()>>;

  /// Get the name of this root metric.
  auto metricName() -> MetricName;

  /// Get the vector of metric names
  auto getMetricNames() -> std::vector<MetricName>;

  /// Insert the supplied time value into the metric storage.
  /// \arg MetricIndex Zero to insert values in the root metric. Index starting with one to insert in the submetric
  /// specified by the index.
  /// \arg Time The time of the time value pair
  /// \arg Value The value of the time value pair
  void insert(uint64_t MetricIndex, std::chrono::high_resolution_clock::time_point Time, double Value);

  /// Insert the supplied time value into the metric storage.
  /// \arg MetricIndex Zero to insert values in the root metric. Index starting with one to insert in the submetric
  /// specified by the index.
  /// \arg TimeSinceEpoch The time since epoch of the time value pair
  /// \arg Value The value of the time value pair
  void insert(uint64_t MetricIndex, int64_t TimeSinceEpoch, double Value);

  /// This function insert a time value pair for a specific metric. This function will be provided to metrics to
  /// allow them to push time value pairs.
  /// \arg MetricIndex Zero to insert values in the root metric. Index starting with one to insert in the submetric
  /// specified by the index.
  /// \arg TimeSinceEpoch The time since epoch of the time value pair
  /// \arg Value The value of the time value pair
  static void insertCallback(void* Cls, uint64_t MetricIndex, int64_t TimeSinceEpoch, double Value) {
    if (!Cls) {
      throw std::invalid_argument("External metric does not provide Cls argument");
    }
    auto& This = *static_cast<RootMetric*>(Cls);
    This.insert(MetricIndex, TimeSinceEpoch, Value);
  }

  /// Get the summaries of the root metric and submetrics between two timepoints with different start and stop delats.
  /// \arg StartTime The start time of the summarized measurement values
  /// \arg StopTime The stop time of the summarized measurement values
  /// \arg StartDelta The time to skip from the measurement start
  /// \arg StopDelta The time to skip from the measurement stop
  /// \arg NumThreads The number of thread the experiment was run wtih.
  /// \returns The map of MetricName to Summmary
  [[nodiscard]] auto getSummaries(std::chrono::high_resolution_clock::time_point StartTime,
                                  std::chrono::high_resolution_clock::time_point StopTime,
                                  std::chrono::milliseconds StartDelta, std::chrono::milliseconds StopDelta,
                                  uint64_t NumThreads) -> MetricSummaries;

private:
  /// Check if the metric is available. This function call init and fini on the metric. If successful, it sets the
  /// available flag and insert the available submetrics into this datastructure.
  void checkAvailability();
};

} // namespace firestarter::measurement
