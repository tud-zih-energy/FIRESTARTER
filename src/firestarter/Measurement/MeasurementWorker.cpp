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

#include <iostream>
#include <queue>
#include <thread>

#ifndef FIRESTARTER_LINK_STATIC
extern "C" {
#include <dlfcn.h>
}
#endif

void insertCallback(void* cls, const char* metricName, int64_t timeSinceEpoch, double value) {
  static_cast<firestarter::measurement::MeasurementWorker*>(cls)->insertCallback(metricName, timeSinceEpoch, value);
}

using namespace firestarter::measurement;

MeasurementWorker::MeasurementWorker(std::chrono::milliseconds updateInterval, uint64_t numThreads,
                                     std::vector<std::string> const& metricDylibs,
                                     std::vector<std::string> const& stdinMetrics)
    : UpdateInterval(updateInterval)
    , NumThreads(numThreads) {

#ifndef FIRESTARTER_LINK_STATIC
  // open dylibs and find metric symbol.
  // create an entry in _metricDylibs with handle from dlopen and
  // metric_interface_t structure. add this structe as a pointer to metrics.
  for (auto const& dylib : metricDylibs) {
    void* handle;
    const char* filename = dylib.c_str();

    handle = dlopen(dylib.c_str(), RTLD_NOW | RTLD_LOCAL);

    if (!handle) {
      firestarter::log::error() << filename << ": " << dlerror();
      continue;
    }

    // clear existing error
    dlerror();

    metric_interface_t* metric = nullptr;

    metric = (metric_interface_t*)dlsym(handle, "metric");

    char* error;
    if ((error = dlerror()) != NULL) {
      firestarter::log::error() << filename << ": " << error;
      dlclose(handle);
      continue;
    }

    if (this->findMetricByName(metric->name) != nullptr) {
      firestarter::log::error() << "A metric named \"" << metric->name << "\" is already loaded.";
      dlclose(handle);
      continue;
    }

    // lets push our metric object and the handle
    this->_metricDylibs.push_back(handle);
    this->metrics.push_back(metric);
  }
#else
  (void)metricDylibs;
#endif

  // setup metric objects for metric names passed from stdin.
  for (auto const& name : stdinMetrics) {
    if (this->findMetricByName(name) != nullptr) {
      firestarter::log::error() << "A metric named \"" << name << "\" is already loaded.";
      continue;
    }

    this->StdinMetrics.push_back(name);
  }

  std::stringstream ss;
  unsigned maxLength = 0;
  std::map<std::string, bool> available;

  for (auto const& metric : this->Metrics) {
    std::string name(metric->Name);
    maxLength = maxLength < name.size() ? name.size() : maxLength;
    int returnCode = metric->Init();
    metric->Fini();
    available[name] = returnCode == EXIT_SUCCESS ? true : false;
  }

  unsigned padding = maxLength > 6 ? maxLength - 6 : 0;
  ss << "  METRIC" << std::string(padding + 1, ' ') << "| available\n";
  ss << "  " << std::string(padding + 7, '-') << "-----------\n";
  for (auto const& [key, value] : available) {
    ss << "  " << key << std::string(padding + 7 - key.size(), ' ') << "| ";
    ss << (value ? "yes" : "no") << "\n";
  }

  this->AvailableMetricsString = ss.str();

  pthread_create(&this->WorkerThread, NULL,
                 reinterpret_cast<void* (*)(void*)>(MeasurementWorker::dataAcquisitionWorker), this);

  // create a worker for getting metric values from stdin
  if (this->StdinMetrics.size() > 0) {
    pthread_create(&this->StdinThread, NULL,
                   reinterpret_cast<void* (*)(void*)>(MeasurementWorker::stdinDataAcquisitionWorker), this);
  }
}

MeasurementWorker::~MeasurementWorker() {
  pthread_cancel(this->WorkerThread);

  pthread_join(this->WorkerThread, NULL);

  if (this->StdinMetrics.size() > 0) {
    pthread_cancel(this->StdinThread);

    pthread_join(this->StdinThread, NULL);
  }

  for (auto const& [key, value] : this->Values) {
    auto metric = this->findMetricByName(key);
    if (metric == nullptr) {
      continue;
    }

    metric->Fini();
  }

#ifndef FIRESTARTER_LINK_STATIC
  for (auto handle : this->_metricDylibs) {
    dlclose(handle);
  }
#endif
}

std::vector<std::string> MeasurementWorker::metricNames() {
  std::vector<std::string> metrics;
  std::transform(this->Metrics.begin(), this->Metrics.end(), std::back_inserter(metrics),
                 [](auto& metric) -> std::string { return std::string(metric->Name); });
  for (auto const& name : this->StdinMetrics) {
    metrics.push_back(name);
  }

  return metrics;
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

  std::vector<std::string> initialized = {};

  // try to find each metric and initialize it
  for (auto const& metricName : MetricNames) {
    // init values map with empty vector
    auto name_equal = [metricName](auto const& pair) { return metricName.compare(pair.first) == 0; };
    auto pair = std::find_if(this->Values.begin(), this->Values.end(), name_equal);
    if (pair != this->Values.end()) {
      pair->second.clear();
    } else {
      auto metric = this->findMetricByName(metricName);
      if (metric != nullptr) {
        int returnValue = metric->Init();
        if (returnValue != EXIT_SUCCESS) {
          log::error() << "Metric " << metric->Name << ": " << metric->GetError();
          continue;
        }
      }
      this->Values[metricName] = std::vector<TimeValue>();
      if (metric != nullptr) {
        if (metric->Type.InsertCallback) {
          metric->RegisterInsertCallback(::insertCallback, this);
        }
      }
      initialized.push_back(metricName);
    }
  }

  this->ValuesMutex.unlock();

  return initialized;
}

void MeasurementWorker::insertCallback(const char* metricName, int64_t timeSinceEpoch, double value) {
  this->ValuesMutex.lock();

  using Duration = std::chrono::duration<int64_t, std::nano>;
  auto time = std::chrono::time_point<std::chrono::high_resolution_clock, Duration>(Duration(timeSinceEpoch));
  auto name_equal = [metricName](auto const& pair) { return std::string(metricName).compare(pair.first) == 0; };
  auto pair = std::find_if(this->Values.begin(), this->Values.end(), name_equal);

  if (pair != this->Values.end()) {
    pair->second.push_back(TimeValue(time, value));
  }

  this->ValuesMutex.unlock();
}

void MeasurementWorker::startMeasurement() { this->StartTime = std::chrono::high_resolution_clock::now(); }

std::map<std::string, Summary> MeasurementWorker::getValues(std::chrono::milliseconds startDelta,
                                                            std::chrono::milliseconds stopDelta) {
  std::map<std::string, Summary> measurment = {};

  this->ValuesMutex.lock();

  for (auto& [key, values] : this->Values) {
    auto startTime = this->StartTime;
    auto endTime = std::chrono::high_resolution_clock::now();
    auto metric = this->findMetricByName(key);

    MetricType type;
    std::memset(&type, 0, sizeof(type));
    if (metric == nullptr) {
      type.Absolute = 1;

      startTime += startDelta;
      endTime -= stopDelta;
    } else {
      std::memcpy(&type, &metric->Type, sizeof(type));

      if (metric->Type.IgnoreStartStopDelta == 0) {
        startTime += startDelta;
        endTime -= stopDelta;
      }
    }

    decltype(values) croppedValues(values.size());

    auto findAll = [startTime, endTime](auto const& tv) { return startTime <= tv.Time && tv.Time <= endTime; };
    auto it = std::copy_if(values.begin(), values.end(), croppedValues.begin(), findAll);
    croppedValues.resize(std::distance(croppedValues.begin(), it));

    Summary sum = Summary::calculate(croppedValues.begin(), croppedValues.end(), type, this->NumThreads);

    measurment[key] = sum;
  }

  this->ValuesMutex.unlock();

  return measurment;
}

int* MeasurementWorker::dataAcquisitionWorker(void* measurementWorker) {

  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

  auto _this = reinterpret_cast<MeasurementWorker*>(measurementWorker);

#ifndef __APPLE__
  pthread_setname_np(pthread_self(), "DataAcquisition");
#endif

  using clock = std::chrono::high_resolution_clock;

  using callbackTuple =
      std::tuple<void (*)(void), std::chrono::microseconds, std::chrono::high_resolution_clock::time_point>;
  auto callbackTupleComparator = [](callbackTuple left, callbackTuple right) {
    return std::get<2>(left) > std::get<2>(right);
  };

  // this datastructure holds a tuple of our callback, the callback frequency
  // and the next timepoint. it will be sorted, so the pop function will give
  // back the next callback
  std::priority_queue<callbackTuple, std::vector<callbackTuple>, decltype(callbackTupleComparator)> callbackQueue(
      callbackTupleComparator);

  _this->ValuesMutex.lock();

  for (auto const& [key, value] : _this->Values) {
    auto metric_interface = _this->findMetricByName(key);

    if (metric_interface == nullptr) {
      continue;
    }

    auto callbackTime = std::chrono::microseconds(metric_interface->CallbackTime);
    if (callbackTime.count() == 0) {
      continue;
    }

    auto currentTime = clock::now();

    callbackQueue.push(std::make_tuple(metric_interface->Callback, callbackTime, currentTime));
  }

  _this->ValuesMutex.unlock();

  auto nextFetch = clock::now() + _this->UpdateInterval;

  for (;;) {
    auto now = clock::now();

    if (nextFetch <= now) {
      _this->ValuesMutex.lock();

      for (auto& [metricName, values] : _this->Values) {
        auto metric_interface = _this->findMetricByName(metricName);

        if (metric_interface == nullptr) {
          continue;
        }

        double value;

        if (!metric_interface->Type.InsertCallback && metric_interface->GetReading != nullptr) {
          if (EXIT_SUCCESS == metric_interface->GetReading(&value)) {
            auto tv = TimeValue(std::chrono::high_resolution_clock::now(), value);
            values.push_back(tv);
          }
        }
      }

      _this->ValuesMutex.unlock();

      nextFetch = now + _this->UpdateInterval;
    }

    auto nextWake = nextFetch;

    if (!callbackQueue.empty()) {
      auto [callbackFunction, callbackTime, nextCallback] = callbackQueue.top();

      if (nextCallback <= now) {
        // remove the elment from the queue
        callbackQueue.pop();

        // call our callback
        callbackFunction();

        // add it with the updated callback time to the queue again
        nextCallback = now + callbackTime;
        callbackQueue.push(std::make_tuple(callbackFunction, callbackTime, nextCallback));
      }

      nextWake = nextCallback < nextWake ? nextCallback : nextWake;
    }

    std::this_thread::sleep_for(nextWake - clock::now());
  }
}

int* MeasurementWorker::stdinDataAcquisitionWorker(void* measurementWorker) {

  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

  auto _this = reinterpret_cast<MeasurementWorker*>(measurementWorker);

#ifndef __APPLE__
  pthread_setname_np(pthread_self(), "StdinDataAcquis");
#endif

  for (std::string line; std::getline(std::cin, line);) {
    int64_t time;
    double value;
    char name[128];
    if (std::sscanf(line.c_str(), "%127s %ld %lf", name, &time, &value) == 3) {
      auto name_equal = [name](auto const& allowedName) { return allowedName.compare(std::string(name)) == 0; };
      auto item = std::find_if(_this->stdinMetrics().begin(), _this->stdinMetrics().end(), name_equal);
      // metric name is allowed
      if (item != _this->stdinMetrics().end()) {
        _this->insertCallback(name, time, value);
      }
    }
  }

  return NULL;
}
