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

#pragma once

#ifdef FIRESTARTER_BUILD_CUDA
#include <firestarter/Cuda/Cuda.hpp>
#endif

#include <firestarter/Constants.hpp>

#if defined(linux) || defined(__linux__)
#include <firestarter/Measurement/MeasurementWorker.hpp>
#include <firestarter/Optimizer/Algorithm.hpp>
#include <firestarter/Optimizer/OptimizerWorker.hpp>
#include <firestarter/Optimizer/Population.hpp>
#endif

#include <firestarter/DumpRegisterWorkerData.hpp>
#include <firestarter/LoadWorkerData.hpp>

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) ||            \
    defined(_M_X64)
#include <firestarter/Environment/X86/X86Environment.hpp>
#endif

#include <chrono>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#if defined(linux) || defined(__linux__)
extern "C" {
#include <pthread.h>
}
#endif

namespace firestarter {

class Firestarter {
public:
  Firestarter(const int argc, const char **argv,
              std::chrono::seconds const &timeout, unsigned loadPercent,
              std::chrono::microseconds const &period,
              unsigned requestedNumThreads, std::string const &cpuBind,
              bool printFunctionSummary, unsigned functionId,
              bool listInstructionGroups, std::string const &instructionGroups,
              unsigned lineCount, bool allowUnavailablePayload,
              bool dumpRegisters,
              std::chrono::seconds const &dumpRegistersTimeDelta,
              std::string const &dumpRegistersOutpath, bool errorDetection,
              int gpus, unsigned gpuMatrixSize, bool gpuUseFloat,
              bool gpuUseDouble, bool listMetrics, bool measurement,
              std::chrono::milliseconds const &startDelta,
              std::chrono::milliseconds const &stopDelta,
              std::chrono::milliseconds const &measurementInterval,
              std::vector<std::string> const &metricPaths,
              std::vector<std::string> const &stdinMetrics, bool optimize,
              std::chrono::seconds const &preheat,
              std::string const &optimizationAlgorithm,
              std::vector<std::string> const &optimizationMetrics,
              std::chrono::seconds const &evaluationDuration,
              unsigned individuals, std::string const &optimizeOutfile,
              unsigned generations, double nsga2_cr, double nsga2_m);

  ~Firestarter();

  void mainThread();

private:
  const int _argc;
  const char **_argv;
  const std::chrono::seconds _timeout;
  const unsigned _loadPercent;
  std::chrono::microseconds _load;
  std::chrono::microseconds _period;
  const bool _dumpRegisters;
  const std::chrono::seconds _dumpRegistersTimeDelta;
  const std::string _dumpRegistersOutpath;
  const bool _errorDetection;
  const int _gpus;
  const unsigned _gpuMatrixSize;
  const bool _gpuUseFloat;
  const bool _gpuUseDouble;
  const std::chrono::milliseconds _startDelta;
  const std::chrono::milliseconds _stopDelta;
  const bool _measurement;
  const bool _optimize;
  const std::chrono::seconds _preheat;
  const std::string _optimizationAlgorithm;
  const std::vector<std::string> _optimizationMetrics;
  const std::chrono::seconds _evaluationDuration;
  const unsigned _individuals;
  const std::string _optimizeOutfile;
  const unsigned _generations;
  const double _nsga2_cr;
  const double _nsga2_m;

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) ||            \
    defined(_M_X64)
  environment::x86::X86Environment *_environment = nullptr;

  environment::x86::X86Environment &environment() const {
    return *_environment;
  }
#else
#error "FIRESTARTER is not implemented for this ISA"
#endif

#ifdef FIRESTARTER_BUILD_CUDA
  std::unique_ptr<cuda::Cuda> _cuda;
#endif

#if defined(linux) || defined(__linux__)
  inline static std::unique_ptr<optimizer::OptimizerWorker> _optimizer;
  std::shared_ptr<measurement::MeasurementWorker> _measurementWorker;
  std::unique_ptr<firestarter::optimizer::Algorithm> _algorithm;
  firestarter::optimizer::Population _population;
#endif

  // LoadThreadWorker.cpp
  int initLoadWorkers(bool lowLoad, unsigned long long period);
  void joinLoadWorkers();
  void printThreadErrorReport();
  void printPerformanceReport();

  void signalWork() { signalLoadWorkers(THREAD_WORK); };

  // WatchdogWorker.cpp
  int watchdogWorker(std::chrono::microseconds period,
                     std::chrono::microseconds load,
                     std::chrono::seconds timeout);

#ifdef FIRESTARTER_DEBUG_FEATURES
  // DumpRegisterWorker.cpp
  int initDumpRegisterWorker(std::chrono::seconds dumpTimeDelta,
                             std::string dumpFilePath);
  void joinDumpRegisterWorker();
#endif

  // LoadThreadWorker.cpp
  void signalLoadWorkers(int comm);
  static void loadThreadWorker(std::shared_ptr<LoadWorkerData> td);

  // CudaWorker.cpp
  static void *cudaWorker(void *cudaData);

#ifdef FIRESTARTER_DEBUG_FEATURES
  // DumpRegisterWorker.cpp
  static void dumpRegisterWorker(std::unique_ptr<DumpRegisterWorkerData> data);
#endif

  static void setLoad(unsigned long long value);

  static void sigalrmHandler(int signum);
  static void sigtermHandler(int signum);

  // variables to control the termination of the watchdog
  inline static bool _watchdog_terminate = false;
  inline static std::condition_variable _watchdogTerminateAlert;
  inline static std::mutex _watchdogTerminateMutex;

  // variable to control the load of the threads
  inline static volatile unsigned long long loadVar = LOAD_LOW;

  std::vector<std::pair<std::thread, std::shared_ptr<LoadWorkerData>>>
      loadThreads;

  std::vector<std::shared_ptr<unsigned long long>> errorCommunication;

#ifdef FIRESTARTER_DEBUG_FEATURES
  std::thread dumpRegisterWorkerThread;
#endif
};

} // namespace firestarter
