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

#pragma once

#if defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP)
#include <firestarter/Cuda/Cuda.hpp>
#endif

#ifdef FIRESTARTER_BUILD_ONEAPI
#include <firestarter/OneAPI/OneAPI.hpp>
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

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
#include <firestarter/Environment/X86/X86Environment.hpp>
#endif

#include <chrono>
#include <condition_variable>
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
  Firestarter(int Argc, const char** Argv, std::chrono::seconds const& Timeout, unsigned LoadPercent,
              std::chrono::microseconds const& Period, unsigned RequestedNumThreads, std::string const& CpuBind,
              bool PrintFunctionSummary, unsigned FunctionId, bool ListInstructionGroups,
              std::string const& InstructionGroups, unsigned LineCount, bool AllowUnavailablePayload,
              bool DumpRegisters, std::chrono::seconds const& DumpRegistersTimeDelta,
              std::string const& DumpRegistersOutpath, bool ErrorDetection, int Gpus, unsigned GpuMatrixSize,
              bool GpuUseFloat, bool GpuUseDouble, bool ListMetrics, bool Measurement,
              std::chrono::milliseconds const& StartDelta, std::chrono::milliseconds const& StopDelta,
              std::chrono::milliseconds const& MeasurementInterval, std::vector<std::string> const& MetricPaths,
              std::vector<std::string> const& StdinMetrics, bool Optimize, std::chrono::seconds const& Preheat,
              std::string const& OptimizationAlgorithm, std::vector<std::string> const& OptimizationMetrics,
              std::chrono::seconds const& EvaluationDuration, unsigned Individuals, std::string const& OptimizeOutfile,
              unsigned Generations, double Nsga2Cr, double Nsga2M);

  ~Firestarter();

  void mainThread();

private:
  const int Argc;
  const char** Argv;
  const std::chrono::seconds Timeout;
  const unsigned LoadPercent;
  std::chrono::microseconds Load;
  std::chrono::microseconds Period;
  const bool DumpRegisters;
  const std::chrono::seconds DumpRegistersTimeDelta;
  const std::string DumpRegistersOutpath;
  const bool ErrorDetection;
  const int Gpus;
  const unsigned GpuMatrixSize;
  const bool GpuUseFloat;
  const bool GpuUseDouble;
  const std::chrono::milliseconds StartDelta;
  const std::chrono::milliseconds StopDelta;
  const bool Measurement;
  const bool Optimize;
  const std::chrono::seconds Preheat;
  const std::string OptimizationAlgorithm;
  const std::vector<std::string> OptimizationMetrics;
  const std::chrono::seconds EvaluationDuration;
  const unsigned Individuals;
  const std::string OptimizeOutfile;
  const unsigned Generations;
  const double Nsga2Cr;
  const double Nsga2M;

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
  environment::x86::X86Environment* Environment = nullptr;

  [[nodiscard]] auto environment() const -> environment::x86::X86Environment& { return *Environment; }
#else
#error "FIRESTARTER is not implemented for this ISA"
#endif

#if defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP)
  std::unique_ptr<cuda::Cuda> _cuda;
#endif

#ifdef FIRESTARTER_BUILD_ONEAPI
  std::unique_ptr<oneapi::OneAPI> _oneapi;
#endif

#if defined(linux) || defined(__linux__)
  inline static std::unique_ptr<optimizer::OptimizerWorker> Optimizer;
  std::shared_ptr<measurement::MeasurementWorker> MeasurementWorker;
  std::unique_ptr<firestarter::optimizer::Algorithm> Algorithm;
  firestarter::optimizer::Population Population;
#endif

  // LoadThreadWorker.cpp
  auto initLoadWorkers(bool LowLoad, uint64_t Period) -> int;
  void joinLoadWorkers();
  void printThreadErrorReport();
  void printPerformanceReport();

  void signalWork() { signalLoadWorkers(THREAD_WORK); };

  // WatchdogWorker.cpp
  auto watchdogWorker(std::chrono::microseconds Period, std::chrono::microseconds Load, std::chrono::seconds Timeout)
      -> int;

#ifdef FIRESTARTER_DEBUG_FEATURES
  // DumpRegisterWorker.cpp
  auto initDumpRegisterWorker(std::chrono::seconds DumpTimeDelta, std::string DumpFilePath) -> int;
  void joinDumpRegisterWorker();
#endif

  // LoadThreadWorker.cpp
  void signalLoadWorkers(int Comm);
  static void loadThreadWorker(std::shared_ptr<LoadWorkerData> Td);

#ifdef FIRESTARTER_DEBUG_FEATURES
  // DumpRegisterWorker.cpp
  static void dumpRegisterWorker(std::unique_ptr<DumpRegisterWorkerData> Data);
#endif

  static void setLoad(uint64_t Value);

  static void sigalrmHandler(int Signum);
  static void sigtermHandler(int Signum);

  // variables to control the termination of the watchdog
  inline static bool WatchdogTerminate = false;
  inline static std::condition_variable WatchdogTerminateAlert;
  inline static std::mutex WatchdogTerminateMutex;

  // variable to control the load of the threads
  inline static volatile uint64_t LoadVar = LOAD_LOW;

  std::vector<std::pair<std::thread, std::shared_ptr<LoadWorkerData>>> LoadThreads;

  std::vector<std::shared_ptr<uint64_t>> ErrorCommunication;

#ifdef FIRESTARTER_DEBUG_FEATURES
  std::thread DumpRegisterWorkerThread;
#endif
};

} // namespace firestarter
