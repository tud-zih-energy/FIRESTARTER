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

#include "firestarter/Logging/Log.hpp"
#include "firestarter/Measurement/MetricInterface.h"
#include "firestarter/Measurement/Summary.hpp"
#include "firestarter/Measurement/TimeValue.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if not(defined(FIRESTARTER_LINK_STATIC)) && defined(linux)
extern "C" {
#include <dlfcn.h>
}
#endif

namespace firestarter::measurement {

static void insertCallback(void* Cls, const char* /*MetricName*/, int64_t TimeSinceEpoch, double Value);

/// This class handels the state around a metric. Its name, the contained time value pairs and a mutex to guard them.
struct Metric {

  Metric() = delete;

  explicit Metric(std::string Name)
      : Name(std::move(Name)) {}

  /// The name of the metric
  std::string Name;
  /// The data collected from the metrics.
  std::vector<TimeValue> Values;
  /// Mutex to access the Values
  std::mutex ValuesMutex;
};

/// This class handels the state around a leaf metric. Its name, the contained time value pairs and a mutex to guard
/// them.
struct LeafMetric : public Metric {
  LeafMetric() = delete;

  explicit LeafMetric(std::string Name)
      : Metric(std::move(Name)) {}
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

  ~RootMetric() {
    if (Initialized && MetricPtr) {
      MetricPtr->Fini();
    }

#if not(defined(FIRESTARTER_LINK_STATIC)) && defined(linux)
    if (Dylib) {
      dlclose(MetricPtr);
    }
#endif
    Initialized = false;
  }

  /// Popuplate the RootMetric object from the C-style MetricInterface.
  /// \arg Metric the refernce to the C-style MetricInterface
  static auto fromCInterface(MetricInterface& Metric) -> std::shared_ptr<RootMetric> {
    auto Root = std::make_shared<RootMetric>(/*Name=*/Metric.Name,
                                             /*Dylib=*/false,
                                             /*Stdin=*/false,
                                             /*Initialized=*/false);
    Root->MetricPtr = &Metric;
    Root->checkAvailability();
    return Root;
  }

#if not(defined(FIRESTARTER_LINK_STATIC)) && defined(linux)
  /// Popuplate the RootMetric object from the C-style MetricInterface provided via a dynamic library.
  /// \arg DylibPath The dynamic library name
  static auto fromDylib(const std::string& DylibPath) -> std::shared_ptr<RootMetric> {
    void* Handle = nullptr;
    const char* Filename = DylibPath.c_str();

    Handle = dlopen(Filename, RTLD_NOW | RTLD_LOCAL);

    if (!Handle) {
      firestarter::log::error() << Filename << ": " << dlerror();
      return nullptr;
    }

    // clear existing error
    dlerror();

    MetricInterface* Metric = nullptr;

    Metric = static_cast<MetricInterface*>(dlsym(Handle, "metric"));

    char* Error = nullptr;
    if ((Error = dlerror()) != nullptr) {
      firestarter::log::error() << Filename << ": " << Error;
      dlclose(Handle);
      return nullptr;
    }

    auto Root = std::make_shared<RootMetric>(/*Name=*/Metric->Name,
                                             /*Dylib=*/true,
                                             /*Stdin=*/false,
                                             /*Initialized=*/false);
    Root->checkAvailability();
    return Root;
  }
#endif

  /// Popuplate the RootMetric object from the C-style MetricInterface.
  /// \arg Metric the refernce to the C-style MetricInterface
  static auto fromStdin(const std::string& MetricName) -> std::shared_ptr<RootMetric> {
    auto Root = std::make_shared<RootMetric>(/*Name=*/MetricName,
                                             /*Dylib=*/false,
                                             /*Stdin=*/true,
                                             /*Initialized=*/true);

    Root->checkAvailability();
    return Root;
  }

  auto initialize() -> bool {
    log::debug() << "Initializing metric " << Name;

    // Clear the contained data
    {
      std::lock_guard<std::mutex> Lock(ValuesMutex);
      Values.clear();
    }
    // Init the associated metric interface object
    if (MetricPtr) {
      const auto ReturnValue = MetricPtr->Init();

      Initialized = ReturnValue == EXIT_SUCCESS;
      if (!Initialized) {
        log::warn() << "Metric " << Name << ": " << MetricPtr->GetError();
        return false;
      }

      // Get the submetrics
      if (MetricPtr->GetSubmetricNames) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        for (auto** Names = MetricPtr->GetSubmetricNames(); Names; Names++) {
          // Create a new submetric entry for each name
          Submetrics.emplace_back(std::make_unique<LeafMetric>(std::string(*Names)));
        }
      }

      // Register the callback to insert data
      if (MetricPtr->Type.InsertCallback) {
        MetricPtr->RegisterInsertCallback(insertCallback, this);
      }
    }

    return true;
  }

  /// Get the callback function and callback interval that has to be executed in a timed fashion for the metric.
  /// \returns an empty optional if the metric does not required timed callbacks. otherwise it returns a tuple of the
  /// callback function and callback interval. The callback function should be called after the callback interval has
  /// expired.
  auto getTimedCallback() -> std::optional<std::tuple<std::function<void()>, std::chrono::microseconds>> {
    if (!MetricPtr) {
      return {};
    }

    auto CallbackTime = std::chrono::microseconds(MetricPtr->CallbackTime);
    if (CallbackTime.count() == 0) {
      return {};
    }

    auto Callback = [this]() {
      if (Initialized) {
        MetricPtr->Callback();
      }
    };

    return std::tuple<std::function<void()>, std::chrono::microseconds>{Callback, CallbackTime};
  }

  /// Get the optional callback that has to be executed to insert values into the metric storage.
  /// \return optionally returns the function that has to be executed to save the current metric value into the
  /// stoarage.
  auto getInsertCallback() -> std::optional<std::function<void()>> {
    if (!MetricPtr) {
      return {};
    }

    auto Callback = [this]() {
      double Value = NAN;

      if (Initialized && EXIT_SUCCESS == MetricPtr->GetReading(&Value)) {
        insert(std::chrono::high_resolution_clock::now(), Value);
      }
    };

    if (!MetricPtr->Type.InsertCallback && MetricPtr->GetReading != nullptr) {
      return Callback;
    }

    return {};
  }

  void insert(std::chrono::high_resolution_clock::time_point Time, double Value) {
    const std::lock_guard<std::mutex> Lock(ValuesMutex);
    Values.emplace_back(Time, Value);
  }

  void insert(int64_t TimeSinceEpoch, double Value) {
    // TODO: allow inserting the sub metric
    using Duration = std::chrono::duration<int64_t, std::nano>;
    auto Time = std::chrono::time_point<std::chrono::high_resolution_clock, Duration>(Duration(TimeSinceEpoch));

    insert(Time, Value);
  }

  [[nodiscard]] auto getSummary(std::chrono::high_resolution_clock::time_point StartTime,
                                std::chrono::high_resolution_clock::time_point StopTime,
                                std::chrono::milliseconds StartDelta, std::chrono::milliseconds StopDelta,
                                const uint64_t NumThreads) -> Summary {
    MetricType Type{};

    if (MetricPtr == nullptr) {
      Type.Absolute = 1;

      StartTime += StartDelta;
      StopTime -= StopDelta;
    } else {
      Type = MetricPtr->Type;

      if (!Type.IgnoreStartStopDelta) {
        StartTime += StartDelta;
        StopTime -= StopDelta;
      }
    }

    decltype(Values) CroppedValues;

    {
      std::lock_guard<std::mutex> Lock(ValuesMutex);

      auto FindAll = [&StartTime, &StopTime](auto const& Tv) { return StartTime <= Tv.Time && Tv.Time <= StopTime; };
      std::copy_if(Values.begin(), Values.end(), std::back_inserter(CroppedValues), FindAll);
    }

    return Summary::calculate(CroppedValues.begin(), CroppedValues.end(), Type, NumThreads);
  }

private:
  void checkAvailability() {
    if (MetricPtr) {
      auto ReturnCode = MetricPtr->Init();
      MetricPtr->Fini();
      Available = ReturnCode == EXIT_SUCCESS;
    } else {
      Available = true;
    }
  }
};

/// This function insert a time value pair for a specific metric. This function will be provided to metrics to
/// allow them to push time value pairs.
/// \arg MetricName The name of the metric for which values are inserted
/// \arg TimeSinceEpoch The time since epoch of the time value pair
/// \arg Value The value of the time value pair
static void insertCallback(void* Cls, const char* /*MetricName*/, int64_t TimeSinceEpoch, double Value) {
  assert(Cls && "External metric does not provide Cls argument");
  auto& This = *static_cast<RootMetric*>(Cls);
  // TODO: allow inserting into the sub metrics
  This.insert(TimeSinceEpoch, Value);
}

} // namespace firestarter::measurement
