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

#include "firestarter/Measurement/Metric.hpp"
#include "firestarter/Config/MetricName.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/Measurement/MetricInterface.h"
#include "firestarter/Measurement/Summary.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <string>
#include <tuple>
#include <vector>

#if not(defined(FIRESTARTER_LINK_STATIC)) && defined(linux)
extern "C" {
#include <dlfcn.h>
}
#endif

namespace firestarter::measurement {

auto Metric::getSummary(const std::chrono::high_resolution_clock::time_point StartTime,
                        const std::chrono::high_resolution_clock::time_point StopTime, const MetricType Type,
                        const uint64_t NumThreads) -> Summary {

  auto FindAll = [&StartTime, &StopTime](auto const& Tv) { return StartTime <= Tv.Time && Tv.Time <= StopTime; };

  decltype(Values) CroppedValues;

  {
    const std::lock_guard<std::mutex> Lock(ValuesMutex);
    std::copy_if(Values.cbegin(), Values.cend(), std::back_inserter(CroppedValues), FindAll);
  }

  return Summary::calculate(CroppedValues.begin(), CroppedValues.end(), Type, NumThreads);
}

auto LeafMetric::metricName() -> MetricName { return {/*Inverted=*/false, RootRef.Name, Name}; }

RootMetric::~RootMetric() {
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

auto RootMetric::fromCInterface(MetricInterface& Metric) -> std::shared_ptr<RootMetric> {
  auto Root = std::make_shared<RootMetric>(/*Name=*/Metric.Name,
                                           /*Dylib=*/false,
                                           /*Stdin=*/false,
                                           /*Initialized=*/false);
  Root->MetricPtr = &Metric;
  Root->checkAvailability();
  return Root;
}

#if not(defined(FIRESTARTER_LINK_STATIC)) && defined(linux)
auto RootMetric::fromDylib(const std::string& DylibPath) -> std::shared_ptr<RootMetric> {
  void* Handle = nullptr;
  const char* Filename = DylibPath.c_str();

  Handle = dlopen(Filename, RTLD_NOW | RTLD_LOCAL);

  if (!Handle) {
    log::error() << Filename << ": " << dlerror();
    return nullptr;
  }

  // clear existing error
  dlerror();

  MetricInterface* Metric = nullptr;

  Metric = static_cast<MetricInterface*>(dlsym(Handle, "metric"));

  char* Error = nullptr;
  if ((Error = dlerror()) != nullptr) {
    log::error() << Filename << ": " << Error;
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

auto RootMetric::fromStdin(const std::string& MetricName) -> std::shared_ptr<RootMetric> {
  auto Root = std::make_shared<RootMetric>(/*Name=*/MetricName,
                                           /*Dylib=*/false,
                                           /*Stdin=*/true,
                                           /*Initialized=*/true);

  Root->checkAvailability();
  return Root;
}

auto RootMetric::initialize() -> bool {
  // Skip if already initialized
  if (Initialized) {
    return true;
  }

  log::debug() << "Initializing metric " << Name;

  // Clear the contained data
  {
    const std::lock_guard<std::mutex> Lock(ValuesMutex);
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

    // Register the callback to insert data
    if (MetricPtr->Type.InsertCallback) {
      MetricPtr->RegisterInsertCallback(insertCallback, this);
    }
  }

  return true;
}

auto RootMetric::getTimedCallback() -> std::optional<std::tuple<std::function<void()>, std::chrono::microseconds>> {
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

auto RootMetric::getInsertCallback() -> std::optional<std::function<void()>> {
  if (!MetricPtr) {
    return {};
  }

  auto Callback = [this]() {
    std::vector<double> Values(1 + Submetrics.size(), NAN);

    if (Initialized && EXIT_SUCCESS == MetricPtr->GetReading(Values.data(), Values.size())) {
      for (auto I = 0U; I < Values.size(); I++) {
        insert(/*MetricIndex=*/I, std::chrono::high_resolution_clock::now(), Values[I]);
      }
    }
  };

  if (!MetricPtr->Type.InsertCallback && MetricPtr->GetReading != nullptr) {
    return Callback;
  }

  return {};
}

auto RootMetric::metricName() -> MetricName { return {/*Inverted=*/false, Name}; }

auto RootMetric::getMetricNames() -> std::vector<MetricName> {
  std::vector<MetricName> MetricNames;
  MetricNames.emplace_back(metricName());

  std::transform(Submetrics.cbegin(), Submetrics.cend(), std::back_inserter(MetricNames),
                 [](const auto& SubMetric) { return SubMetric->metricName(); });

  return MetricNames;
}

void RootMetric::insert(uint64_t MetricIndex, std::chrono::high_resolution_clock::time_point Time, double Value) {
  const std::lock_guard<std::mutex> Lock(ValuesMutex);

  if (MetricIndex == ROOT_METRIC_INDEX) {
    Values.emplace_back(Time, Value);
  } else {
    auto Index = MetricIndex - 1;
    if (Index < Submetrics.size()) {
      Submetrics[Index]->Values.emplace_back(Time, Value);
    }
  }
}

void RootMetric::insert(uint64_t MetricIndex, int64_t TimeSinceEpoch, double Value) {
  using Duration = std::chrono::duration<int64_t, std::nano>;
  auto Time = std::chrono::time_point<std::chrono::high_resolution_clock, Duration>(Duration(TimeSinceEpoch));

  insert(MetricIndex, Time, Value);
}

auto RootMetric::getSummaries(std::chrono::high_resolution_clock::time_point StartTime,
                              std::chrono::high_resolution_clock::time_point StopTime,
                              std::chrono::milliseconds StartDelta, std::chrono::milliseconds StopDelta,
                              const uint64_t NumThreads) -> MetricSummaries {
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

  MetricSummaries Summaries;

  Summaries[metricName()] = getSummary(StartTime, StopTime, Type, NumThreads);

  for (const auto& Submetric : Submetrics) {
    Summaries[Submetric->metricName()] = Submetric->getSummary(StartTime, StopTime, Type, NumThreads);
  }

  return Summaries;
}

void RootMetric::checkAvailability() {
  if (MetricPtr) {
    auto ReturnCode = MetricPtr->Init();
    Available = ReturnCode == EXIT_SUCCESS;

    // Get the submetrics
    if (Available && MetricPtr->GetSubmetricNames) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      for (auto** Names = MetricPtr->GetSubmetricNames(); *Names; Names++) {
        // Create a new submetric entry for each name
        Submetrics.emplace_back(std::make_unique<LeafMetric>(std::string(*Names), *this));
      }
    }

    MetricPtr->Fini();

  } else {
    Available = true;
  }
}

} // namespace firestarter::measurement