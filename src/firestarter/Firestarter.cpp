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
#if defined(linux) || defined(__linux__)
#include <firestarter/Optimizer/Algorithm/NSGA2.hpp>
#include <firestarter/Optimizer/History.hpp>
#include <firestarter/Optimizer/Problem/CLIArgumentProblem.hpp>
#endif

#include <csignal>
#include <functional>
#include <thread>

using namespace firestarter;

Firestarter::Firestarter(
    const int argc, const char **argv, std::chrono::seconds const &timeout,
    unsigned loadPercent, std::chrono::microseconds const &period,
    unsigned requestedNumThreads, std::string const &cpuBind,
    bool printFunctionSummary, unsigned functionId, bool listInstructionGroups,
    std::string const &instructionGroups, unsigned lineCount,
    bool allowUnavailablePayload, bool dumpRegisters,
    std::chrono::seconds const &dumpRegistersTimeDelta,
    std::string const &dumpRegistersOutpath, int gpus, unsigned gpuMatrixSize,
    bool gpuUseFloat, bool gpuUseDouble, bool listMetrics, bool measurement,
    std::chrono::milliseconds const &startDelta,
    std::chrono::milliseconds const &stopDelta,
    std::chrono::milliseconds const &measurementInterval,
    std::vector<std::string> const &metricPaths,
    std::vector<std::string> const &stdinMetrics, bool optimize,
    std::chrono::seconds const &preheat,
    std::string const &optimizationAlgorithm,
    std::vector<std::string> const &optimizationMetrics,
    std::chrono::seconds const &evaluationDuration, unsigned individuals,
    std::string const &optimizeOutfile, unsigned generations, double nsga2_cr,
    double nsga2_m)
    : _argc(argc), _argv(argv), _timeout(timeout), _loadPercent(loadPercent),
      _period(period), _dumpRegisters(dumpRegisters),
      _dumpRegistersTimeDelta(dumpRegistersTimeDelta),
      _dumpRegistersOutpath(dumpRegistersOutpath), _startDelta(startDelta),
      _stopDelta(stopDelta), _measurement(measurement), _optimize(optimize),
      _preheat(preheat), _optimizationAlgorithm(optimizationAlgorithm),
      _optimizationMetrics(optimizationMetrics),
      _evaluationDuration(evaluationDuration), _individuals(individuals),
      _optimizeOutfile(optimizeOutfile), _generations(generations),
      _nsga2_cr(nsga2_cr), _nsga2_m(nsga2_m) {
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
  if (_measurement || listMetrics || _optimize) {
    _measurementWorker = std::make_shared<measurement::MeasurementWorker>(
        measurementInterval, this->environment().requestedNumThreads(),
        metricPaths, stdinMetrics);

    if (listMetrics) {
      log::info() << _measurementWorker->availableMetrics();
      std::exit(EXIT_SUCCESS);
    }

    // init all metrics
    auto all = _measurementWorker->metricNames();
    auto initialized = _measurementWorker->initMetrics(all);

    if (initialized.size() == 0) {
      log::error() << "No metrics initialized";
      std::exit(EXIT_FAILURE);
    }

    // check if selected metrics are initialized
    for (auto const &optimizationMetric : optimizationMetrics) {
      auto nameEqual = [optimizationMetric](auto const &name) {
        return name.compare(optimizationMetric) == 0;
      };
      // metric name is not found
      if (std::find_if(all.begin(), all.end(), nameEqual) == all.end()) {
        log::error() << "Metric \"" << optimizationMetric
                     << "\" does not exist.";
        std::exit(EXIT_FAILURE);
      }
      // metric has not initialized properly
      if (std::find_if(initialized.begin(), initialized.end(), nameEqual) ==
          initialized.end()) {
        log::error() << "Metric \"" << optimizationMetric
                     << "\" failed to initialize.";
        std::exit(EXIT_FAILURE);
      }
    }
  }

  if (_optimize) {
    std::vector<std::string> instructionGroups = {};
    for (auto const &[key, value] :
         this->environment().selectedConfig().payloadSettings()) {
      instructionGroups.push_back(key);
    }
    auto applySettings = std::bind(
        [this](std::vector<std::pair<std::string, unsigned>> const &setting) {
          using Clock = std::chrono::high_resolution_clock;
          auto start = Clock::now();

          for (auto &thread : this->loadThreads) {
            auto td = thread.second;

            td->config().setPayloadSettings(setting);
          }

          for (auto const &thread : this->loadThreads) {
            auto td = thread.second;

            td->mutex.lock();
          }

          for (auto const &thread : this->loadThreads) {
            auto td = thread.second;

            td->comm = THREAD_SWITCH;
            td->mutex.unlock();
          }

          this->loadVar = LOAD_SWITCH;

          for (auto const &thread : this->loadThreads) {
            auto td = thread.second;
            bool ack;

            do {
              td->mutex.lock();
              ack = td->ack;
              td->mutex.unlock();
            } while (!ack);

            td->mutex.lock();
            td->ack = false;
            td->mutex.unlock();
          }

          this->loadVar = LOAD_HIGH;

          this->signalWork();

          auto end = Clock::now();

          log::trace() << "Switching payload took "
                       << std::chrono::duration_cast<std::chrono::milliseconds>(
                              end - start)
                              .count()
                       << "ms";
        },
        std::placeholders::_1);

    auto prob =
        std::make_shared<firestarter::optimizer::problem::CLIArgumentProblem>(
            std::move(applySettings), _measurementWorker, _optimizationMetrics,
            _evaluationDuration, _startDelta, _stopDelta, instructionGroups);

    _population = firestarter::optimizer::Population(std::move(prob));

    if (_optimizationAlgorithm == "NSGA2") {
      _algorithm = std::make_unique<firestarter::optimizer::algorithm::NSGA2>(
          _generations, _nsga2_cr, _nsga2_m);
    } else {
      throw std::invalid_argument("Algorithm " + _optimizationAlgorithm +
                                  " unknown.");
    }

    _algorithm->checkPopulation(
        static_cast<firestarter::optimizer::Population const &>(_population),
        _individuals);
  }
#endif

  this->environment().printSelectedCodePathSummary();

  log::info() << this->environment().topology();

  // setup thread with either high or low load configured at the start
  // low loads has to know the length of the period
  if (EXIT_SUCCESS !=
      (returnCode = this->initLoadWorkers((_loadPercent == 0), _period.count(),
                                          _dumpRegisters))) {
    std::exit(returnCode);
  }

  // add some signal handler for aborting FIRESTARTER
#ifndef _WIN32
  std::signal(SIGALRM, Firestarter::sigalrmHandler);
#endif

  std::signal(SIGTERM, Firestarter::sigtermHandler);
  std::signal(SIGINT, Firestarter::sigtermHandler);
}

Firestarter::~Firestarter() {
#ifdef FIRESTARTER_BUILD_CUDA
  free(this->gpuStructPointer);
#endif

  delete _environment;
}

void Firestarter::mainThread() {
  int returnCode;

  this->environment().printThreadSummary();

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
  if (_measurement) {
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

#if defined(linux) || defined(__linux__)
  // check if optimization is selected
  if (_optimize) {
    auto startTime = optimizer::History::getTime();

    Firestarter::_optimizer = std::make_unique<optimizer::OptimizerWorker>(
        std::move(_algorithm), _population, _optimizationAlgorithm,
        _individuals, _preheat);

    // wait here until optimizer thread terminates
    Firestarter::_optimizer->join();

    firestarter::optimizer::History::save(_optimizeOutfile, startTime, _argc,
                                          _argv);

    // stop all the load threads
    std::raise(SIGTERM);
  }
#endif

  // wait for watchdog to timeout or until user terminates
  this->joinLoadWorkers();
#ifdef FIRESTARTER_DEBUG_FEATURES
  if (_dumpRegisters) {
    this->joinDumpRegisterWorker();
  }
#endif

  if (!_optimize) {
    this->printPerformanceReport();
  }

#if defined(linux) || defined(__linux__)
  // if measurment is enabled, stop it here
  if (_measurement) {
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

void Firestarter::setLoad(unsigned long long value) {
  // signal load change to workers
  Firestarter::loadVar = value;
  __asm__ __volatile__("mfence;");
}

void Firestarter::sigalrmHandler(int signum) { (void)signum; }

void Firestarter::sigtermHandler(int signum) {
  (void)signum;

  // required for cases load = {0,100}, which do no enter the loop
  Firestarter::setLoad(LOAD_STOP);
  // exit loop
  // used in case of 0 < load < 100
  Firestarter::_watchdog_terminate = true;

#if defined(linux) || defined(__linux__)
  // if we have optimization running stop it
  if (Firestarter::_optimizer) {
    Firestarter::_optimizer->kill();
  }
#endif
}
