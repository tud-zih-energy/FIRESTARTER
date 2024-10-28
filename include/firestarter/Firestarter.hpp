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

#include "Config.hpp"
#include "Constants.hpp"
#include "Cuda/Cuda.hpp"
#include "DumpRegisterWorkerData.hpp"
#include "LoadWorkerData.hpp"
#include "Measurement/MeasurementWorker.hpp"
#include "OneAPI/OneAPI.hpp"
#include "Optimizer/Algorithm.hpp"
#include "Optimizer/OptimizerWorker.hpp"
#include "Optimizer/Population.hpp"

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

  explicit Firestarter(Config&& ProvidedConfig);

  ~Firestarter() = default;

  void mainThread();

private:
  const Config Cfg;

  std::unique_ptr<environment::Environment> Environment;
  std::unique_ptr<cuda::Cuda> Cuda;
  std::unique_ptr<oneapi::OneAPI> Oneapi;
  std::unique_ptr<firestarter::optimizer::Algorithm> Algorithm;
  std::thread DumpRegisterWorkerThread;
  std::shared_ptr<measurement::MeasurementWorker> MeasurementWorker;

  std::vector<std::pair<std::thread, std::shared_ptr<LoadWorkerData>>> LoadThreads;
  std::vector<std::shared_ptr<uint64_t>> ErrorCommunication;

  firestarter::optimizer::Population Population;

  inline static std::unique_ptr<optimizer::OptimizerWorker> Optimizer;

  // variables to control the termination of the watchdog
  inline static bool WatchdogTerminate = false;
  inline static std::condition_variable WatchdogTerminateAlert;
  inline static std::mutex WatchdogTerminateMutex;

  // variable to control the load of the threads
  inline static volatile LoadThreadWorkType LoadVar = LoadThreadWorkType::LoadLow;

  // LoadThreadWorker.cpp
  void initLoadWorkers();
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

      Td->config().settings().selectInstructionGroups(Setting);
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

  // DumpRegisterWorker.cpp
  void initDumpRegisterWorker();
  void joinDumpRegisterWorker();
  static void dumpRegisterWorker(std::unique_ptr<DumpRegisterWorkerData> Data);

  static void setLoad(LoadThreadWorkType Value);

  static void sigalrmHandler(int Signum);
  static void sigtermHandler(int Signum);
};

} // namespace firestarter
