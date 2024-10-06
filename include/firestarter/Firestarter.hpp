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
#include "Cuda/Cuda.hpp"
#endif

#ifdef FIRESTARTER_BUILD_ONEAPI
#include "OneAPI/OneAPI.hpp"
#endif

#include "Constants.hpp"

#if defined(linux) || defined(__linux__)
#include "Measurement/MeasurementWorker.hpp"
#include "Optimizer/Algorithm.hpp"
#include "Optimizer/OptimizerWorker.hpp"
#include "Optimizer/Population.hpp"
#endif

#include "DumpRegisterWorkerData.hpp"
#include "LoadWorkerData.hpp"

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
  Firestarter() = delete;

  Firestarter(int Argc, const char** Argv, std::chrono::seconds const& Timeout, unsigned LoadPercent,
              std::chrono::microseconds const& Period, unsigned RequestedNumThreads, std::string const& CpuBind,
              bool PrintFunctionSummary, unsigned FunctionId, bool ListInstructionGroups,
              std::string const& InstructionGroups, unsigned LineCount, bool AllowUnavailablePayload,
              bool DumpRegisters, std::chrono::seconds const& DumpRegistersTimeDelta, std::string DumpRegistersOutpath,
              bool ErrorDetection, int Gpus, unsigned GpuMatrixSize, bool GpuUseFloat, bool GpuUseDouble,
              bool ListMetrics, bool Measurement, std::chrono::milliseconds const& StartDelta,
              std::chrono::milliseconds const& StopDelta, std::chrono::milliseconds const& MeasurementInterval,
              std::vector<std::string> const& MetricPaths, std::vector<std::string> const& StdinMetrics, bool Optimize,
              std::chrono::seconds const& Preheat, std::string const& OptimizationAlgorithm,
              std::vector<std::string> const& OptimizationMetrics, std::chrono::seconds const& EvaluationDuration,
              unsigned Individuals, std::string OptimizeOutfile, unsigned Generations, double Nsga2Cr, double Nsga2M);

  ~Firestarter() = default;

  void mainThread();

private:
  const int Argc;
  const char** Argv;
  const std::chrono::seconds Timeout;
  const unsigned LoadPercent;
  std::chrono::microseconds Load{};
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

  std::unique_ptr<environment::Environment> Environment;

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
  void initLoadWorkers(bool LowLoad, std::chrono::microseconds Period);
  void joinLoadWorkers();
  void printThreadErrorReport();
  void printPerformanceReport();

  /// Set the load workers to the ThreadInit state.
  void signalInit() { signalLoadWorkers(LoadThreadState::ThreadInit); }

  /// Set the load workers to the ThreadWork state.
  void signalWork() { signalLoadWorkers(LoadThreadState::ThreadWork); };

  /// Set the load workers to the ThreadWork state.
  /// \arg Setting The new setting to switch to.
  void signalSwitch(std::vector<std::pair<std::string, unsigned>> const& Setting) {
    struct SwitchLoad {
      static void func() { LoadVar = LoadThreadWorkType::LoadSwitch; };
    };

    for (auto& Thread : LoadThreads) {
      auto Td = Thread.second;

      Td->config().setPayloadSettings(Setting);
    }

    signalLoadWorkers(LoadThreadState::ThreadSwitch, SwitchLoad::func);
  };

  /// Execute a state change in the load worker threads. This should happen at the same time in all threads. First the
  /// mutex in all threads are locked an then the state is updated and we wait until we get an acknowledgement from the
  /// threads.
  /// \arg State The new state of the threads.
  /// \arg Function An optional function that will be executed after the state in all threads has been updated and
  /// before we wait for the acknowledgement of the thread.
  void signalLoadWorkers(LoadThreadState State, void (*Function)() = nullptr);

  static void loadThreadWorker(const std::shared_ptr<LoadWorkerData>& Td);

  // WatchdogWorker.cpp
  static auto watchdogWorker(std::chrono::microseconds Period, std::chrono::microseconds Load,
                             std::chrono::seconds Timeout) -> int;

#ifdef FIRESTARTER_DEBUG_FEATURES
  // DumpRegisterWorker.cpp
  void initDumpRegisterWorker(std::chrono::seconds DumpTimeDelta, const std::string& DumpFilePath);
  void joinDumpRegisterWorker();
  static void dumpRegisterWorker(std::unique_ptr<DumpRegisterWorkerData> Data);

  std::thread DumpRegisterWorkerThread;
#endif

  static void setLoad(LoadThreadWorkType Value);

  static void sigalrmHandler(int Signum);
  static void sigtermHandler(int Signum);

  // variables to control the termination of the watchdog
  inline static bool WatchdogTerminate = false;
  inline static std::condition_variable WatchdogTerminateAlert;
  inline static std::mutex WatchdogTerminateMutex;

  // variable to control the load of the threads
  inline static volatile LoadThreadWorkType LoadVar = LoadThreadWorkType::LoadLow;

  std::vector<std::pair<std::thread, std::shared_ptr<LoadWorkerData>>> LoadThreads;

  std::vector<std::shared_ptr<uint64_t>> ErrorCommunication;
};

} // namespace firestarter
