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

#include "firestarter/Measurement/MeasurementWorker.hpp"
#include "firestarter/Config/MetricName.hpp"
#include "firestarter/Measurement/Metric.hpp"
#include "firestarter/Measurement/MetricInterface.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

// NOLINTBEGIN(cert-dcl50-cpp,cppcoreguidelines-pro-type-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,clang-analyzer-valist.Uninitialized)
auto scanStdin(const char* Fmt, int Count, ...) -> bool {
  va_list Args;
  va_start(Args, Count);
  auto ReturnCode = std::vscanf(Fmt, Args);
  va_end(Args);
  return ReturnCode == Count;
}
// NOLINTEND(cert-dcl50-cpp,cppcoreguidelines-pro-type-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,clang-analyzer-valist.Uninitialized)

} // namespace

namespace firestarter::measurement {

MeasurementWorker::MeasurementWorker(std::chrono::milliseconds UpdateInterval, uint64_t NumThreads,
                                     std::vector<std::string> const& MetricDylibsNames,
                                     std::vector<std::string> const& StdinMetricsNames)
    : UpdateInterval(UpdateInterval)
    , NumThreads(NumThreads) {
#if not(defined(FIRESTARTER_LINK_STATIC)) && defined(linux)
  for (auto const& Dylib : MetricDylibsNames) {
    Metrics.emplace_back(RootMetric::fromDylib(Dylib));
  }
#else
  (void)MetricDylibsNames;
#endif

  // setup metric objects for metric names passed from stdin.
  for (auto const& Name : StdinMetricsNames) {
    Metrics.emplace_back(RootMetric::fromStdin(Name));
  }

  // If the size of the vector and its set matches length, we don't have duplicate names.
  const std::vector<MetricName> MetricNamesVec = metrics();
  const auto MetricNamesSet = std::set<MetricName>(MetricNamesVec.cbegin(), MetricNamesVec.cend());

  if (MetricNamesVec.size() != MetricNamesSet.size()) {
    throw std::invalid_argument("Duplicate metric names present.");
  }

  WorkerThread = std::thread(MeasurementWorker::dataAcquisitionWorker, std::ref(*this));

  if (StdinMetricsNames.size() > 1) {
    // create a worker for getting metric values from stdin
    StdinThread = std::thread(MeasurementWorker::stdinDataAcquisitionWorker, std::ref(*this));
  }
}

MeasurementWorker::~MeasurementWorker() {
  StopExecution.store(true);
  if (WorkerThread.joinable()) {
    WorkerThread.join();
  }
  if (WorkerThread.joinable()) {
    StdinThread.detach();
  }
}

// this must be called by the main thread.
// if not done so things like perf_event_attr.inherit might not work as expected
void MeasurementWorker::initMetrics(std::vector<MetricName> const& MetricNames) {
  // try to find each metric and initialize it
  for (auto const& MetricName : MetricNames) {
    const auto Metric = findRootMetricByName(MetricName);
    if (!Metric) {
      throw std::invalid_argument("Could not find metric: " + MetricName.toString());
    }

    Metric->initialize();
  }
}

void MeasurementWorker::startMeasurement() { StartTime = std::chrono::high_resolution_clock::now(); }

auto MeasurementWorker::getValues(std::chrono::milliseconds StartDelta,
                                  std::chrono::milliseconds StopDelta) -> MetricSummaries {
  MetricSummaries Measurement;

  auto StartTime = this->StartTime;
  auto StopTime = std::chrono::high_resolution_clock::now();

  for (auto& Metric : Metrics) {
    if (Metric->Initialized) {
      Measurement.merge(Metric->getSummaries(StartTime, StopTime, StartDelta, StopDelta, NumThreads));
    }
  }

  return Measurement;
}

void MeasurementWorker::dataAcquisitionWorker(MeasurementWorker& This) {
#if defined(linux) || defined(__linux__)
  // NOLINTNEXTLINE(misc-include-cleaner)
  pthread_setname_np(pthread_self(), "DataAcquisition");
#endif

  using Clock = std::chrono::high_resolution_clock;

  using CallbackTuple =
      std::tuple<std::function<void()>, std::chrono::microseconds, std::chrono::high_resolution_clock::time_point>;
  auto CallbackTupleComparator = [](CallbackTuple Left, CallbackTuple Right) {
    return std::get<2>(Left) > std::get<2>(Right);
  };

  // this datastructure holds a tuple of our callback, the callback frequency
  // and the next timepoint. it will be sorted, so the pop function will give
  // back the next callback
  std::priority_queue<CallbackTuple, std::vector<CallbackTuple>, decltype(CallbackTupleComparator)> CallbackQueue(
      CallbackTupleComparator);

  auto StartTime = Clock::now();

  // Register the periodic callbacks
  for (auto const& Metric : This.Metrics) {
    // Register the periodic callbacks for metric internal use
    {
      auto OptionalTimedCallback = Metric->getTimedCallback();

      if (OptionalTimedCallback) {
        const auto& [Callback, CallbackTime] = *OptionalTimedCallback;
        CallbackQueue.emplace(Callback, CallbackTime, StartTime);
      }
    }

    // Register the periodic callback that pull the metric value.
    {
      auto OptionalCallback = Metric->getInsertCallback();
      if (OptionalCallback) {
        CallbackQueue.emplace(*OptionalCallback, This.UpdateInterval, StartTime);
      }
    }
  }

  while (!This.StopExecution.load()) {
    auto Now = Clock::now();

    // Default 1us timeout for no work.
    auto SleepTime = std::chrono::microseconds(1);

    if (!CallbackQueue.empty()) {
      // Get the next callback and execute if the time matches. Once executed, reregister.
      {
        auto [CallbackFunction, CallbackTime, NextCallback] = CallbackQueue.top();

        if (NextCallback <= Now) {
          // remove the elment from the queue
          CallbackQueue.pop();

          // call our callback
          CallbackFunction();

          // add it with the updated callback time to the queue again
          NextCallback = Now + CallbackTime;
          CallbackQueue.emplace(CallbackFunction, CallbackTime, NextCallback);
        }
      }

      // Adjust the sleep time to the next callback
      {
        auto [CallbackFunction, CallbackTime, NextCallback] = CallbackQueue.top();
        SleepTime = std::chrono::duration_cast<std::chrono::microseconds>(NextCallback - Clock::now());
        // Sleep time may not be negative
        SleepTime = (std::max)(std::chrono::microseconds(0), SleepTime);
      }
    }

    std::this_thread::sleep_for(SleepTime);
  }
}

void MeasurementWorker::stdinDataAcquisitionWorker(MeasurementWorker& This) {

#if defined(linux) || defined(__linux__)
  // NOLINTNEXTLINE(misc-include-cleaner)
  pthread_setname_np(pthread_self(), "StdinDataAcquis");
#endif

  for (;;) {
    int64_t Time = 0;
    double Value = NAN;
    std::array<char, 128> Name = {0};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    if (!scanStdin("%127s %ld %lf", 3, Name.data(), &Time, &Value)) {
      continue;
    }

    const auto Metric = This.findRootMetricByName(MetricName(/*Inverted=*/false, Name.data()));
    if (Metric) {
      Metric->insert(ROOT_METRIC_INDEX, Time, Value);
    }
  }
}

} // namespace firestarter::measurement