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

#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>
#if defined(linux) || defined(__linux__)
#include <firestarter/Optimizer/Algorithm/NSGA2.hpp>
#include <firestarter/Optimizer/History.hpp>
#include <firestarter/Optimizer/Problem/CLIArgumentProblem.hpp>
extern "C" {
#include <firestarter/Measurement/Metric/IPCEstimate.h>
}
#endif

#include <csignal>
#include <functional>
#include <thread>

#ifdef _MSC_VER
#include <intrin.h>
#endif

using namespace firestarter;

Firestarter::Firestarter(const int Argc, const char** Argv, std::chrono::seconds const& Timeout, unsigned LoadPercent,
                         std::chrono::microseconds const& Period, unsigned RequestedNumThreads,
                         std::string const& CpuBind, bool PrintFunctionSummary, unsigned FunctionId,
                         bool ListInstructionGroups, std::string const& InstructionGroups, unsigned LineCount,
                         bool AllowUnavailablePayload, bool DumpRegisters,
                         std::chrono::seconds const& DumpRegistersTimeDelta, std::string const& DumpRegistersOutpath,
                         bool ErrorDetection, int Gpus, unsigned GpuMatrixSize, bool GpuUseFloat, bool GpuUseDouble,
                         bool ListMetrics, bool Measurement, std::chrono::milliseconds const& StartDelta,
                         std::chrono::milliseconds const& StopDelta,
                         std::chrono::milliseconds const& MeasurementInterval,
                         std::vector<std::string> const& MetricPaths, std::vector<std::string> const& StdinMetrics,
                         bool Optimize, std::chrono::seconds const& Preheat, std::string const& OptimizationAlgorithm,
                         std::vector<std::string> const& OptimizationMetrics,
                         std::chrono::seconds const& EvaluationDuration, unsigned Individuals,
                         std::string const& OptimizeOutfile, unsigned Generations, double Nsga2Cr, double Nsga2M)
    : Argc(Argc)
    , Argv(Argv)
    , Timeout(Timeout)
    , LoadPercent(LoadPercent)
    , Period(Period)
    , DumpRegisters(DumpRegisters)
    , DumpRegistersTimeDelta(DumpRegistersTimeDelta)
    , DumpRegistersOutpath(DumpRegistersOutpath)
    , ErrorDetection(ErrorDetection)
    , Gpus(Gpus)
    , GpuMatrixSize(GpuMatrixSize)
    , GpuUseFloat(GpuUseFloat)
    , GpuUseDouble(GpuUseDouble)
    , StartDelta(StartDelta)
    , StopDelta(StopDelta)
    , Measurement(Measurement)
    , Optimize(Optimize)
    , Preheat(Preheat)
    , OptimizationAlgorithm(OptimizationAlgorithm)
    , OptimizationMetrics(OptimizationMetrics)
    , EvaluationDuration(EvaluationDuration)
    , Individuals(Individuals)
    , OptimizeOutfile(OptimizeOutfile)
    , Generations(Generations)
    , Nsga2Cr(Nsga2Cr)
    , Nsga2M(Nsga2M) {
  int returnCode;

  Load = (Period * LoadPercent) / 100;
  if (LoadPercent == 100 || Load == std::chrono::microseconds::zero()) {
    this->Period = std::chrono::microseconds::zero();
  }

#if defined(linux) || defined(__linux__)
#else
  (void)ListMetrics;
  (void)MeasurementInterval;
  (void)MetricPaths;
  (void)StdinMetrics;
#endif

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
  this->Environment = new environment::x86::X86Environment();
#endif

  if (EXIT_SUCCESS != (returnCode = this->environment().evaluateCpuAffinity(RequestedNumThreads, CpuBind))) {
    std::exit(returnCode);
  }

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
  // Error detection uses crc32 instruction added by the SSE4.2 extension to x86
  if (ErrorDetection) {
    if (!Environment->topology().featuresAsmjit().has(asmjit::CpuFeatures::X86::kSSE4_2)) {
      throw std::invalid_argument("Option --error-detection requires the crc32 "
                                  "instruction added with SSE_4_2.\n");
    }
  }
#endif

  if (ErrorDetection && this->environment().requestedNumThreads() < 2) {
    throw std::invalid_argument("Option --error-detection must run with 2 or more threads. Number of "
                                "threads is " +
                                std::to_string(this->environment().requestedNumThreads()) + "\n");
  }

  this->environment().evaluateFunctions();

  if (PrintFunctionSummary) {
    this->environment().printFunctionSummary();
    std::exit(EXIT_SUCCESS);
  }

  if (EXIT_SUCCESS != (returnCode = this->environment().selectFunction(FunctionId, AllowUnavailablePayload))) {
    std::exit(returnCode);
  }

  if (ListInstructionGroups) {
    this->environment().printAvailableInstructionGroups();
    std::exit(EXIT_SUCCESS);
  }

  if (!InstructionGroups.empty()) {
    if (EXIT_SUCCESS != (returnCode = this->environment().selectInstructionGroups(InstructionGroups))) {
      std::exit(returnCode);
    }
  }

  if (LineCount != 0) {
    this->environment().setLineCount(LineCount);
  }

#if defined(linux) || defined(__linux__)
  if (Measurement || ListMetrics || Optimize) {
    MeasurementWorker = std::make_shared<measurement::MeasurementWorker>(
        MeasurementInterval, this->environment().requestedNumThreads(), MetricPaths, StdinMetrics);

    if (ListMetrics) {
      log::info() << MeasurementWorker->availableMetrics();
      std::exit(EXIT_SUCCESS);
    }

    // init all metrics
    auto all = MeasurementWorker->metricNames();
    auto initialized = MeasurementWorker->initMetrics(all);

    if (initialized.size() == 0) {
      log::error() << "No metrics initialized";
      std::exit(EXIT_FAILURE);
    }

    // check if selected metrics are initialized
    for (auto const& optimizationMetric : OptimizationMetrics) {
      auto nameEqual = [optimizationMetric](auto const& name) {
        auto invertedName = "-" + name;
        return name.compare(optimizationMetric) == 0 || invertedName.compare(optimizationMetric) == 0;
      };
      // metric name is not found
      if (std::find_if(all.begin(), all.end(), nameEqual) == all.end()) {
        log::error() << "Metric \"" << optimizationMetric << "\" does not exist.";
        std::exit(EXIT_FAILURE);
      }
      // metric has not initialized properly
      if (std::find_if(initialized.begin(), initialized.end(), nameEqual) == initialized.end()) {
        log::error() << "Metric \"" << optimizationMetric << "\" failed to initialize.";
        std::exit(EXIT_FAILURE);
      }
    }
  }

  if (Optimize) {
    auto applySettings = std::bind(
        [this](std::vector<std::pair<std::string, unsigned>> const& setting) {
          using Clock = std::chrono::high_resolution_clock;
          auto start = Clock::now();

          for (auto& thread : this->LoadThreads) {
            auto td = thread.second;

            td->config().setPayloadSettings(setting);
          }

          for (auto const& thread : this->LoadThreads) {
            auto td = thread.second;

            td->Mutex.lock();
          }

          for (auto const& thread : this->LoadThreads) {
            auto td = thread.second;

            td->Comm = THREAD_SWITCH;
            td->Mutex.unlock();
          }

          this->LoadVar = LOAD_SWITCH;

          for (auto const& thread : this->LoadThreads) {
            auto td = thread.second;
            bool ack;

            do {
              td->Mutex.lock();
              ack = td->Ack;
              td->Mutex.unlock();
            } while (!ack);

            td->Mutex.lock();
            td->Ack = false;
            td->Mutex.unlock();
          }

          this->LoadVar = LOAD_HIGH;

          this->signalWork();

          uint64_t startTimestamp = 0xffffffffffffffff;
          uint64_t stopTimestamp = 0;

          for (auto const& thread : this->LoadThreads) {
            auto td = thread.second;

            if (startTimestamp > td->LastStartTsc) {
              startTimestamp = td->LastStartTsc;
            }
            if (stopTimestamp < td->LastStopTsc) {
              stopTimestamp = td->LastStopTsc;
            }
          }

          for (auto const& thread : this->LoadThreads) {
            auto td = thread.second;
            ipcEstimateMetricInsert((double)td->LastIterations *
                                    (double)this->LoadThreads.front().second->config().payload().instructions() /
                                    (double)(stopTimestamp - startTimestamp));
          }

          auto end = Clock::now();

          log::trace() << "Switching payload took "
                       << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms";
        },
        std::placeholders::_1);

    auto prob = std::make_shared<firestarter::optimizer::problem::CLIArgumentProblem>(
        std::move(applySettings), MeasurementWorker, OptimizationMetrics, EvaluationDuration, StartDelta, StopDelta,
        this->environment().selectedConfig().payloadItems());

    Population = firestarter::optimizer::Population(std::move(prob));

    if (OptimizationAlgorithm == "NSGA2") {
      Algorithm = std::make_unique<firestarter::optimizer::algorithm::NSGA2>(Generations, Nsga2Cr, Nsga2M);
    } else {
      throw std::invalid_argument("Algorithm " + OptimizationAlgorithm + " unknown.");
    }

    Algorithm->checkPopulation(static_cast<firestarter::optimizer::Population const&>(Population), Individuals);
  }
#endif

  this->environment().printSelectedCodePathSummary();

  log::info() << this->environment().topology();

  // setup thread with either high or low load configured at the start
  // low loads has to know the length of the period
  if (EXIT_SUCCESS != (returnCode = this->initLoadWorkers((LoadPercent == 0), Period.count()))) {
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
#if defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP)
  _cuda.reset();
#endif
#ifdef FIRESTARTER_BUILD_ONEAPI
  _oneapi.reset();
#endif

  delete Environment;
}

void Firestarter::mainThread() {
  this->environment().printThreadSummary();

#if defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP)
  _cuda = std::make_unique<cuda::Cuda>(&this->loadVar, _gpuUseFloat, _gpuUseDouble, _gpuMatrixSize, _gpus);
#endif

#ifdef FIRESTARTER_BUILD_ONEAPI
  _oneapi = std::make_unique<oneapi::OneAPI>(&this->loadVar, _gpuUseFloat, _gpuUseDouble, _gpuMatrixSize, _gpus);
#endif

#if defined(linux) || defined(__linux__)
  // if measurement is enabled, start it here
  if (Measurement) {
    MeasurementWorker->startMeasurement();
  }
#endif

  this->signalWork();

#ifdef FIRESTARTER_DEBUG_FEATURES
  if (DumpRegisters) {
    int returnCode;
    if (EXIT_SUCCESS != (returnCode = this->initDumpRegisterWorker(DumpRegistersTimeDelta, DumpRegistersOutpath))) {
      std::exit(returnCode);
    }
  }
#endif

  // worker thread for load control
  this->watchdogWorker(Period, Load, Timeout);

#if defined(linux) || defined(__linux__)
  // check if optimization is selected
  if (Optimize) {
    auto startTime = optimizer::History::getTime();

    Firestarter::Optimizer = std::make_unique<optimizer::OptimizerWorker>(std::move(Algorithm), Population,
                                                                          OptimizationAlgorithm, Individuals, Preheat);

    // wait here until optimizer thread terminates
    Firestarter::Optimizer->join();

    auto payloadItems = this->environment().selectedConfig().payloadItems();

    firestarter::optimizer::History::save(OptimizeOutfile, startTime, payloadItems, Argc, Argv);

    // print the best 20 according to each metric
    firestarter::optimizer::History::printBest(OptimizationMetrics, payloadItems);

    // stop all the load threads
    std::raise(SIGTERM);
  }
#endif

  // wait for watchdog to timeout or until user terminates
  this->joinLoadWorkers();
#ifdef FIRESTARTER_DEBUG_FEATURES
  if (DumpRegisters) {
    this->joinDumpRegisterWorker();
  }
#endif

  if (!Optimize) {
    this->printPerformanceReport();
  }

#if defined(linux) || defined(__linux__)
  // if measurment is enabled, stop it here
  if (Measurement) {
    // TODO: clear this up
    log::info() << "metric,num_timepoints,duration_ms,average,stddev";
    for (auto const& [name, sum] : MeasurementWorker->getValues(StartDelta, StopDelta)) {
      log::info() << std::quoted(name) << "," << sum.NumTimepoints << "," << sum.Duration.count() << "," << sum.Average
                  << "," << sum.Stddev;
    }
  }
#endif

  if (ErrorDetection) {
    this->printThreadErrorReport();
  }
}

void Firestarter::setLoad(uint64_t value) {
  // signal load change to workers
  Firestarter::LoadVar = value;
#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
#ifndef _MSC_VER
  __asm__ __volatile__("mfence;");
#else
  _mm_mfence();
#endif
#else
#error "FIRESTARTER is not implemented for this ISA"
#endif
}

void Firestarter::sigalrmHandler(int signum) { (void)signum; }

void Firestarter::sigtermHandler(int signum) {
  (void)signum;

  Firestarter::setLoad(LOAD_STOP);
  // exit loop
  // used in case of 0 < load < 100
  // or interrupt sleep for timeout
  {
    std::lock_guard<std::mutex> lk(Firestarter::WatchdogTerminateMutex);
    Firestarter::WatchdogTerminate = true;
  }
  Firestarter::WatchdogTerminateAlert.notify_all();

#if defined(linux) || defined(__linux__)
  // if we have optimization running stop it
  if (Firestarter::Optimizer) {
    Firestarter::Optimizer->kill();
  }
#endif
}
