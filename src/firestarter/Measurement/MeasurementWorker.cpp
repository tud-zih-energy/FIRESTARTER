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

#include <firestarter/Logging/Log.hpp>
#include <firestarter/Measurement/MeasurementWorker.hpp>
#include <queue>

#ifndef FIRESTARTER_LINK_STATIC
extern "C" {
#include <dlfcn.h>
}
#endif

void insertCallback(void* Cls, const char* MetricName, int64_t TimeSinceEpoch, double Value) {
  static_cast<firestarter::measurement::MeasurementWorker*>(Cls)->insertCallback(MetricName, TimeSinceEpoch, Value);
}

namespace firestarter::measurement {

MeasurementWorker::MeasurementWorker(std::chrono::milliseconds UpdateInterval, uint64_t NumThreads,
                                     std::vector<std::string> const& MetricDylibs,
                                     std::vector<std::string> const& StdinMetrics)
    : UpdateInterval(UpdateInterval)
    , NumThreads(NumThreads) {

#ifndef FIRESTARTER_LINK_STATIC
  // open dylibs and find metric symbol.
  // create an entry in _metricDylibs with handle from dlopen and
  // metric_interface_t structure. add this structe as a pointer to metrics.
  for (auto const& Dylib : MetricDylibs) {
    void* Handle = nullptr;
    const char* Filename = Dylib.c_str();

    Handle = dlopen(Dylib.c_str(), RTLD_NOW | RTLD_LOCAL);

    if (!Handle) {
      firestarter::log::error() << Filename << ": " << dlerror();
      continue;
    }

    // clear existing error
    dlerror();

    MetricInterface* Metric = nullptr;

    Metric = static_cast<MetricInterface*>(dlsym(Handle, "metric"));

    char* Error = nullptr;
    if ((Error = dlerror()) != nullptr) {
      firestarter::log::error() << Filename << ": " << Error;
      dlclose(Handle);
      continue;
    }

    if (this->findMetricByName(Metric->Name) != nullptr) {
      firestarter::log::error() << "A metric named \"" << Metric->Name << "\" is already loaded.";
      dlclose(Handle);
      continue;
    }

    // lets push our metric object and the handle
    this->MetricDylibs.push_back(Handle);
    this->Metrics.push_back(Metric);
  }
#else
  (void)MetricDylibs;
#endif

  // setup metric objects for metric names passed from stdin.
  for (auto const& Name : StdinMetrics) {
    if (this->findMetricByName(Name) != nullptr) {
      firestarter::log::error() << "A metric named \"" << Name << "\" is already loaded.";
      continue;
    }

    this->StdinMetrics.push_back(Name);
  }

  std::stringstream Ss;
  unsigned MaxLength = 0;
  std::map<std::string, bool> Available;

  for (auto const& Metric : this->Metrics) {
    std::string Name(Metric->Name);
    MaxLength = MaxLength < Name.size() ? Name.size() : MaxLength;
    auto ReturnCode = Metric->Init();
    Metric->Fini();
    Available[Name] = ReturnCode == EXIT_SUCCESS;
  }

  unsigned Padding = MaxLength > 6 ? MaxLength - 6 : 0;
  Ss << "  METRIC" << std::string(Padding + 1, ' ') << "| available\n";
  Ss << "  " << std::string(Padding + 7, '-') << "-----------\n";
  for (auto const& [key, value] : Available) {
    Ss << "  " << key << std::string(Padding + 7 - key.size(), ' ') << "| ";
    Ss << (value ? "yes" : "no") << "\n";
  }

  this->AvailableMetricsString = Ss.str();

  pthread_create(&this->WorkerThread, nullptr,
                 reinterpret_cast<void* (*)(void*)>(MeasurementWorker::dataAcquisitionWorker), this);

  // create a worker for getting metric values from stdin
  if (this->StdinMetrics.size() > 0) {
    pthread_create(&this->StdinThread, nullptr,
                   reinterpret_cast<void* (*)(void*)>(MeasurementWorker::stdinDataAcquisitionWorker), this);
  }
}

MeasurementWorker::~MeasurementWorker() {
  pthread_cancel(this->WorkerThread);

  pthread_join(this->WorkerThread, nullptr);

  if (this->StdinMetrics.size() > 0) {
    pthread_cancel(this->StdinThread);

    pthread_join(this->StdinThread, nullptr);
  }

  for (auto const& [key, value] : this->Values) {
    const auto* Metric = this->findMetricByName(key);
    if (Metric == nullptr) {
      continue;
    }

    Metric->Fini();
  }

#ifndef FIRESTARTER_LINK_STATIC
  for (auto* Handle : this->MetricDylibs) {
    dlclose(Handle);
  }
#endif
}

auto MeasurementWorker::metricNames() -> std::vector<std::string> {
  std::vector<std::string> Metrics;
  std::transform(this->Metrics.begin(), this->Metrics.end(), std::back_inserter(Metrics),
                 [](auto& Metric) -> std::string { return std::string(Metric->Name); });
  for (auto const& Name : this->StdinMetrics) {
    Metrics.push_back(Name);
  }

  return Metrics;
}

auto MeasurementWorker::findMetricByName(std::string MetricName) -> const MetricInterface* {
  auto NameEqual = [MetricName](auto& MetricInterface) { return MetricName.compare(MetricInterface->Name) == 0; };
  auto Metric = std::find_if(this->Metrics.begin(), this->Metrics.end(), NameEqual);

  // metric not found
  if (Metric == this->Metrics.end()) {
    return nullptr;
  }
  // metric found
  return const_cast<const MetricInterface*>(*Metric);
}

// this must be called by the main thread.
// if not done so things like perf_event_attr.inherit might not work as expected
auto MeasurementWorker::initMetrics(std::vector<std::string> const& MetricNames) -> std::vector<std::string> {
  this->ValuesMutex.lock();

  std::vector<std::string> Initialized = {};

  // try to find each metric and initialize it
  for (auto const& MetricName : MetricNames) {
    // init values map with empty vector
    auto NameEqual = [MetricName](auto const& Pair) { return MetricName.compare(Pair.first) == 0; };
    auto Pair = std::find_if(this->Values.begin(), this->Values.end(), NameEqual);
    if (Pair != this->Values.end()) {
      Pair->second.clear();
    } else {
      const auto* Metric = this->findMetricByName(MetricName);
      if (Metric != nullptr) {
        int ReturnValue = Metric->Init();
        if (ReturnValue != EXIT_SUCCESS) {
          log::error() << "Metric " << Metric->Name << ": " << Metric->GetError();
          continue;
        }
      }
      this->Values[MetricName] = std::vector<TimeValue>();
      if (Metric != nullptr) {
        if (Metric->Type.InsertCallback) {
          Metric->RegisterInsertCallback(::insertCallback, this);
        }
      }
      Initialized.push_back(MetricName);
    }
  }

  this->ValuesMutex.unlock();

  return Initialized;
}

void MeasurementWorker::insertCallback(const char* MetricName, int64_t TimeSinceEpoch, double Value) {
  this->ValuesMutex.lock();

  using Duration = std::chrono::duration<int64_t, std::nano>;
  auto Time = std::chrono::time_point<std::chrono::high_resolution_clock, Duration>(Duration(TimeSinceEpoch));
  auto NameEqual = [MetricName](auto const& Pair) { return std::string(MetricName).compare(Pair.first) == 0; };
  auto Pair = std::find_if(this->Values.begin(), this->Values.end(), NameEqual);

  if (Pair != this->Values.end()) {
    Pair->second.emplace_back(Time, Value);
  }

  this->ValuesMutex.unlock();
}

void MeasurementWorker::startMeasurement() { this->StartTime = std::chrono::high_resolution_clock::now(); }

auto MeasurementWorker::getValues(std::chrono::milliseconds StartDelta, std::chrono::milliseconds StopDelta)
    -> std::map<std::string, Summary> {
  std::map<std::string, Summary> Measurment = {};

  this->ValuesMutex.lock();

  for (auto& [key, values] : this->Values) {
    auto StartTime = this->StartTime;
    auto EndTime = std::chrono::high_resolution_clock::now();
    const auto* Metric = this->findMetricByName(key);

    MetricType Type;
    std::memset(&Type, 0, sizeof(Type));
    if (Metric == nullptr) {
      Type.Absolute = 1;

      StartTime += StartDelta;
      EndTime -= StopDelta;
    } else {
      std::memcpy(&Type, &Metric->Type, sizeof(Type));

      if (Metric->Type.IgnoreStartStopDelta == 0) {
        StartTime += StartDelta;
        EndTime -= StopDelta;
      }
    }

    decltype(values) CroppedValues(values.size());

    auto FindAll = [StartTime, EndTime](auto const& Tv) { return StartTime <= Tv.Time && Tv.Time <= EndTime; };
    auto It = std::copy_if(values.begin(), values.end(), CroppedValues.begin(), FindAll);
    CroppedValues.resize(std::distance(CroppedValues.begin(), It));

    Summary Sum = Summary::calculate(CroppedValues.begin(), CroppedValues.end(), Type, this->NumThreads);

    Measurment[key] = Sum;
  }

  this->ValuesMutex.unlock();

  return Measurment;
}

auto MeasurementWorker::dataAcquisitionWorker(void* MeasurementWorker) -> int* {

  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);

  auto* This = reinterpret_cast<class MeasurementWorker*>(MeasurementWorker);

#ifndef __APPLE__
  pthread_setname_np(pthread_self(), "DataAcquisition");
#endif

  using Clock = std::chrono::high_resolution_clock;

  using CallbackTuple =
      std::tuple<void (*)(void), std::chrono::microseconds, std::chrono::high_resolution_clock::time_point>;
  auto CallbackTupleComparator = [](CallbackTuple Left, CallbackTuple Right) {
    return std::get<2>(Left) > std::get<2>(Right);
  };

  // this datastructure holds a tuple of our callback, the callback frequency
  // and the next timepoint. it will be sorted, so the pop function will give
  // back the next callback
  std::priority_queue<CallbackTuple, std::vector<CallbackTuple>, decltype(CallbackTupleComparator)> CallbackQueue(
      CallbackTupleComparator);

  This->ValuesMutex.lock();

  for (auto const& [key, value] : This->Values) {
    const auto* MetricInterface = This->findMetricByName(key);

    if (MetricInterface == nullptr) {
      continue;
    }

    auto CallbackTime = std::chrono::microseconds(MetricInterface->CallbackTime);
    if (CallbackTime.count() == 0) {
      continue;
    }

    auto CurrentTime = Clock::now();

    CallbackQueue.emplace(MetricInterface->Callback, CallbackTime, CurrentTime);
  }

  This->ValuesMutex.unlock();

  auto NextFetch = Clock::now() + This->UpdateInterval;

  for (;;) {
    auto Now = Clock::now();

    if (NextFetch <= Now) {
      This->ValuesMutex.lock();

      for (auto& [metricName, values] : This->Values) {
        const auto* MetricInterface = This->findMetricByName(metricName);

        if (MetricInterface == nullptr) {
          continue;
        }

        double Value = NAN;

        if (!MetricInterface->Type.InsertCallback && MetricInterface->GetReading != nullptr) {
          if (EXIT_SUCCESS == MetricInterface->GetReading(&Value)) {
            auto Tv = TimeValue(std::chrono::high_resolution_clock::now(), Value);
            values.push_back(Tv);
          }
        }
      }

      This->ValuesMutex.unlock();

      NextFetch = Now + This->UpdateInterval;
    }

    auto NextWake = NextFetch;

    if (!CallbackQueue.empty()) {
      auto [callbackFunction, callbackTime, nextCallback] = CallbackQueue.top();

      if (nextCallback <= Now) {
        // remove the elment from the queue
        CallbackQueue.pop();

        // call our callback
        callbackFunction();

        // add it with the updated callback time to the queue again
        nextCallback = Now + callbackTime;
        CallbackQueue.emplace(callbackFunction, callbackTime, nextCallback);
      }

      NextWake = nextCallback < NextWake ? nextCallback : NextWake;
    }

    std::this_thread::sleep_for(NextWake - Clock::now());
  }
}

auto MeasurementWorker::stdinDataAcquisitionWorker(void* MeasurementWorker) -> int* {

  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);

  auto* This = reinterpret_cast<class MeasurementWorker*>(MeasurementWorker);

#ifndef __APPLE__
  pthread_setname_np(pthread_self(), "StdinDataAcquis");
#endif

  for (std::string Line; std::getline(std::cin, Line);) {
    int64_t Time = 0;
    double Value = NAN;
    char Name[128];
    if (std::sscanf(Line.c_str(), "%127s %ld %lf", Name, &Time, &Value) == 3) {
      auto NameEqual = [Name](auto const& AllowedName) { return AllowedName.compare(std::string(Name)) == 0; };
      auto Item = std::find_if(This->stdinMetrics().begin(), This->stdinMetrics().end(), NameEqual);
      // metric name is allowed
      if (Item != This->stdinMetrics().end()) {
        This->insertCallback(Name, Time, Value);
      }
    }
  }

  return nullptr;
}

} // namespace firestarter::measurement