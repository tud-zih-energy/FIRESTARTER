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

#include "firestarter/Config/Config.hpp"
#include "firestarter/Constants.hpp"
#include "firestarter/Cuda/Cuda.hpp"
#include "firestarter/DumpRegisterWorkerData.hpp"
#include "firestarter/Environment/ThreadAffinity.hpp"
#include "firestarter/LoadWorkerData.hpp"
#include "firestarter/Measurement/MeasurementWorker.hpp"
#include "firestarter/OneAPI/OneAPI.hpp"
#include "firestarter/Optimizer/Algorithm.hpp"
#include "firestarter/Optimizer/OptimizerWorker.hpp"
#include "firestarter/Optimizer/Population.hpp"

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

/// This is the main class of firestarter and handles the execution of the programm.
class Firestarter {
public:
  Firestarter() = delete;

  /// Read the config, validate and throw on problems with config. Setup everything that is required for the execution
  /// of firestarter.
  /// \arg ProvidedConfig The config for the execution of Firestarter
  explicit Firestarter(Config&& ProvidedConfig);

  ~Firestarter() = default;

  /// This function takes care of the execution of firestarter. It will start the load on CPUs and GPUs.
  void mainThread();

private:
  const Config Cfg;

  /// This class handles getting the topology information of the processor and is used to set thread binding.
  std::unique_ptr<environment::CPUTopology> Topology;
  /// The class that handles setting up the payload for firestarter
  std::unique_ptr<environment::Environment> Environment;
  /// The class for execution of the gemm routine on Cuda or HIP GPUs.
  std::unique_ptr<cuda::Cuda> Cuda;
  /// The class for execution of the gemm routine on OneAPI GPUs.
  std::unique_ptr<oneapi::OneAPI> Oneapi;
  /// The pointer to the optimization algorithm that is used by the optimization functionality.
  std::unique_ptr<firestarter::optimizer::Algorithm> Algorithm;
  /// The thread that is used to dump register contents to a file.
  std::thread DumpRegisterWorkerThread;
  /// The shared pointer to the datastructure that handles the management of metrics, acquisition of metric data and
  /// provids summaries of a time range of metric values.
  std::shared_ptr<measurement::MeasurementWorker> MeasurementWorker;

  /// The vector of thread handles for the load workers and shared pointer to the their respective data.
  std::vector<std::pair<std::thread, std::shared_ptr<LoadWorkerData>>> LoadThreads;
  /// The vector of communication data, where each element is shared between two neighbouring threads for the error
  /// detection feature.
  std::vector<std::shared_ptr<uint64_t>> ErrorCommunication;

  /// The population holding the problem that is used for the optimization feature.
  std::unique_ptr<firestarter::optimizer::Population> Population;

  // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
  // TODO(Issue #85): Currently we support one instance of the Firestarter class. Variables that need to be accessed
  // from outside the class, e.g. in the sigterm handler are inline static.

  /// The instance of the optimization worker that handles the execution of the optimization.
  inline static std::unique_ptr<optimizer::OptimizerWorker> Optimizer;

  /// Variable to control the termination of the watchdog
  inline static bool WatchdogTerminate = false;
  /// Condition variable for the WatchdogTerminate to allow notifying when sleeping for a specific time.
  inline static std::condition_variable WatchdogTerminateAlert;
  /// Mutex to guard access to WatchdogTerminate.
  inline static std::mutex WatchdogTerminateMutex;

  /// Variable to control the load of the threads
  inline static volatile LoadThreadWorkType LoadVar = LoadThreadWorkType::LoadLow;

  // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

  /// Spawn the load workers and initialize them.
  /// \arg Affinity Describes the number of threads and how they should be bound to the CPUs.
  void initLoadWorkers(const environment::ThreadAffinity& Affinity);

  /// Wait for the load worker to join
  void joinLoadWorkers();

  /// Print the error report for the error detection feature.
  void printThreadErrorReport();

  /// Print the performance report. It contains the estimation of the FLOPS and main memory bandwidth.
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

  /// The function that is executed for each load thread.
  /// \arg Td The shared pointer to the data that is required in this thread.
  static void loadThreadWorker(const std::shared_ptr<LoadWorkerData>& Td);

  /// This function handels switching the load from high to low in a loop and stopping the execution if a timeout was
  /// set.
  /// \arg Period The period of the high/low switch. Set to zero to disable switching between a high and low load.
  /// \arg Load The time of the period where high load is applied.
  /// \arg Timeout The timeout after which firestarter stops. Set to zero to disable.
  static void watchdogWorker(std::chrono::microseconds Period, std::chrono::microseconds Load,
                             std::chrono::seconds Timeout);

  /// Start the thread to dump the registers of the first load thread to a file.
  void initDumpRegisterWorker();

  /// Wait for the dump register thread to terminate.
  void joinDumpRegisterWorker();

  /// The thread that dumps the registers of the first thread to a file.
  /// \arg Data The data that is required for the worker thread to dump the register contents to a file.
  static void dumpRegisterWorker(std::unique_ptr<DumpRegisterWorkerData> Data);

  /// Set the load var to a specific value and update it with a memory fence across threads.
  /// \arg Value The new load value.
  static void setLoad(LoadThreadWorkType Value);

  /// Sigalarm handler does nothing.
  static void sigalrmHandler(int Signum);

  /// Sigterm handler stops the execution of firestarter
  /// \arg Signum The signal number is ignored.
  static void sigtermHandler(int Signum);
};

} // namespace firestarter
