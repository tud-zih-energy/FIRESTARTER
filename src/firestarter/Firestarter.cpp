/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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

#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>

#include <thread>

using namespace firestarter;

Firestarter::Firestarter(
    std::chrono::seconds timeout, unsigned loadPercent,
    std::chrono::microseconds period, unsigned requestedNumThreads,
    std::string cpuBind, bool printFunctionSummary, unsigned functionId,
    bool listInstructionGroups, std::string instructionGroups,
    unsigned lineCount, bool allowUnavailablePayload, bool dumpRegisters,
    std::chrono::seconds dumpRegistersTimeDelta,
    std::string dumpRegistersOutpath, int gpus, unsigned gpuMatrixSize,
    bool gpuUseFloat, bool gpuUseDouble, bool listMetrics, bool measurement,
    std::chrono::milliseconds startDelta, std::chrono::milliseconds stopDelta,
    std::chrono::milliseconds measurementInterval,
    std::vector<std::string> metricPaths)
    : _timeout(timeout), _loadPercent(loadPercent), _period(period),
      _dumpRegisters(dumpRegisters),
      _dumpRegistersTimeDelta(dumpRegistersTimeDelta),
      _dumpRegistersOutpath(dumpRegistersOutpath), _startDelta(startDelta),
      _stopDelta(stopDelta) {
  int returnCode;

  _load = (_period * _loadPercent) / 100;
  if (_loadPercent == 100 || _load == std::chrono::microseconds::zero()) {
    _period = std::chrono::microseconds::zero();
  }

#ifdef FIRESTARTER_BUILD_CUDA
  this->_gpuStructPointer =
      reinterpret_cast<cuda::gpustruct_t *>(malloc(sizeof(cuda::gpustruct_t)));
  this->_gpuStructPointer->loadingdone = 0;
  this->_gpuStructPointer->loadvar = &this->loadVar;

  if (useGpuFloat) {
    this->gpuStructPointer->use_double = 0;
  } else if (useGpuDouble) {
    this->gpuStructPointer->use_double = 1;
  } else {
    this->gpuStructPointer->use_double = 2;
  }

  this->gpuStructPointer->msize = matrixSize;

  this->gpuStructPointer->use_device = gpus;
#else
  (void)gpus;
  (void)gpuMatrixSize;
  (void)gpuUseFloat;
  (void)gpuUseDouble;
#endif

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) ||            \
    defined(_M_X64)
  this->_environment = new environment::x86::X86Environment();
#endif

  if (EXIT_SUCCESS != (returnCode = this->environment().evaluateCpuAffinity(
                           requestedNumThreads, cpuBind))) {
    std::exit(returnCode);
  }

  this->environment().evaluateFunctions();

  if (printFunctionSummary) {
    this->environment().printFunctionSummary();
    std::exit(EXIT_SUCCESS);
  }

  if (EXIT_SUCCESS != (returnCode = this->environment().selectFunction(
                           functionId, allowUnavailablePayload))) {
    std::exit(returnCode);
  }

  if (listInstructionGroups) {
    this->environment().printAvailableInstructionGroups();
    std::exit(EXIT_SUCCESS);
  }

  if (!instructionGroups.empty()) {
    if (EXIT_SUCCESS !=
        (returnCode =
             this->environment().selectInstructionGroups(instructionGroups))) {
      std::exit(returnCode);
    }
  }

  if (lineCount != 0) {
    this->environment().setLineCount(lineCount);
  }

#if defined(linux) || defined(__linux__)
  if (measurement || listMetrics) {
    _measurementWorker = new measurement::MeasurementWorker(
        measurementInterval, this->environment().requestedNumThreads(),
        metricPaths);

    if (listMetrics) {
      log::info() << _measurementWorker->availableMetrics();
      delete _measurementWorker;
      std::exit(EXIT_SUCCESS);
    }

    // TODO: select the metrics
    // init all metrics
    auto count =
        _measurementWorker->initMetrics(_measurementWorker->metricNames());

    if (count == 0) {
      log::error() << "No metrics initialized";
      delete _measurementWorker;
      std::exit(EXIT_FAILURE);
    }
  }
#endif

  this->environment().printSelectedCodePathSummary();

  log::info() << this->environment().topology();
}

Firestarter::~Firestarter() {
#ifdef FIRESTARTER_BUILD_CUDA
  free(this->gpuStructPointer);
#endif

#if defined(linux) || defined(__linux__)
  if (_measurementWorker != nullptr) {
    delete _measurementWorker;
  }
#endif

  delete _environment;
}

void Firestarter::mainThread() {
  int returnCode;

  this->environment().printThreadSummary();

  // setup thread with either high or low load configured at the start
  // low loads has to know the length of the period
  if (EXIT_SUCCESS !=
      (returnCode = this->initLoadWorkers((_loadPercent == 0), _period.count(),
                                          _dumpRegisters))) {
    std::exit(returnCode);
  }

#ifdef FIRESTARTER_BUILD_CUDA
  pthread_t gpu_thread;
  pthread_create(&gpu_thread, NULL, cuda::init_gpu,
                 (void *)this->gpuStructPointer);
  while (this->gpuStructPointer->loadingdone != 1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
#endif

#if defined(linux) || defined(__linux__)
  // if measurement is enabled, start it here
  if (nullptr != _measurementWorker) {
    _measurementWorker->startMeasurement();
  }
#endif

  this->signalWork();

#ifdef FIRESTARTER_DEBUG_FEATURES
  if (_dumpRegisters) {
    if (EXIT_SUCCESS != (returnCode = this->initDumpRegisterWorker(
                             _dumpRegistersTimeDelta, _dumpRegistersOutpath))) {
      std::exit(returnCode);
    }
  }
#endif

  // worker thread for load control
  this->watchdogWorker(_period, _load, _timeout);

  // wait for watchdog to timeout or until user terminates
  this->joinLoadWorkers();
#ifdef FIRESTARTER_DEBUG_FEATURES
  if (_dumpRegisters) {
    this->joinDumpRegisterWorker();
  }
#endif

  this->printPerformanceReport();

#if defined(linux) || defined(__linux__)
  // if measurment is enabled, stop it here
  if (nullptr != _measurementWorker) {
    // TODO: clear this up
    log::info() << "metric,num_timepoints,duration_ms,average,stddev";
    for (auto const &[name, sum] :
         _measurementWorker->getValues(_startDelta, _stopDelta)) {
      log::info() << std::quoted(name) << "," << sum.num_timepoints << ","
                  << sum.duration.count() << "," << sum.average << ","
                  << sum.stddev;
    }
  }
#endif
}
