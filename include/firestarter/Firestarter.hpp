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

#if defined(linux) || defined(__linux__)
#include <firestarter/Measurement/MeasurementWorker.hpp>
#endif

#include <firestarter/DumpRegisterWorkerData.hpp>
#include <firestarter/LoadWorkerData.hpp>

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) ||            \
    defined(_M_X64)
#include <firestarter/Environment/X86/X86Environment.hpp>
#endif

#include <chrono>
#include <list>
#include <string>
#include <utility>

extern "C" {
#include <pthread.h>
}

namespace firestarter {

class Firestarter {
public:
  Firestarter(std::chrono::seconds const &timeout, unsigned loadPercent,
              std::chrono::microseconds const &period,
              unsigned requestedNumThreads, std::string const &cpuBind,
              bool printFunctionSummary, unsigned functionId,
              bool listInstructionGroups, std::string const &instructionGroups,
              unsigned lineCount, bool allowUnavailablePayload,
              bool dumpRegisters,
              std::chrono::seconds const &dumpRegistersTimeDelta,
              std::string const &dumpRegistersOutpath, int gpus,
              unsigned gpuMatrixSize, bool gpuUseFloat, bool gpuUseDouble,
              bool listMetrics, bool measurement,
              std::chrono::milliseconds const &startDelta,
              std::chrono::milliseconds const &stopDelta,
              std::chrono::milliseconds const &measurementInterval,
              std::vector<std::string> const &metricPaths,
              std::vector<std::string> const &stdinMetrics);

  ~Firestarter();

  void mainThread();

private:
  std::chrono::seconds _timeout;
  unsigned _loadPercent;
  std::chrono::microseconds _load;
  std::chrono::microseconds _period;
  bool _dumpRegisters;
  std::chrono::seconds _dumpRegistersTimeDelta;
  std::string _dumpRegistersOutpath;
  std::chrono::milliseconds _startDelta;
  std::chrono::milliseconds _stopDelta;

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
  cuda::gpustruct_t *const &gpuStructPointer = _gpuStructPointer;
#endif

#if defined(linux) || defined(__linux__)
  measurement::MeasurementWorker *_measurementWorker = nullptr;
#endif

  // LoadThreadWorker.cpp
  int initLoadWorkers(bool lowLoad, unsigned long long period,
                      bool dumpRegisters);
  void joinLoadWorkers();
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

#ifdef FIRESTARTER_BUILD_CUDA
  cuda::gpustruct_t *_gpuStructPointer;
#endif

  // LoadThreadWorker.cpp
  void signalLoadWorkers(int comm);
  static void *loadThreadWorker(void *loadWorkerData);

  // CudaWorker.cpp
  static void *cudaWorker(void *cudaData);

#ifdef FIRESTARTER_DEBUG_FEATURES
  // DumpRegisterWorker.cpp
  static void *dumpRegisterWorker(void *dumpRegisterWorkerData);
#endif

  // variable to control the load of the threads
  volatile unsigned long long loadVar = LOAD_LOW;

  std::list<std::pair<pthread_t *, LoadWorkerData *>> loadThreads;

#ifdef FIRESTARTER_DEBUG_FEATURES
  pthread_t dumpRegisterWorkerThread;
#endif
};

} // namespace firestarter
