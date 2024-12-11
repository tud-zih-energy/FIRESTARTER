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

#include "firestarter/CPUTopology.hpp"
#include "firestarter/Constants.hpp"
#include "firestarter/LoadWorkerMemory.hpp"
#include "firestarter/Platform/PlatformConfig.hpp"
#include "firestarter/ProcessorInformation.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

namespace firestarter {

/// This class contains the information that is required to execute the load routines and change the payload during
/// executions.
class LoadWorkerData {
public:
  /// This struct models parameters acquired during the execution of the high-load routine.
  struct Metrics {
    /// The number of iteration the high-load loop was executed.
    std::atomic<uint64_t> Iterations{};
    /// The start of the execution of the high-load loop.
    std::atomic<uint64_t> StartTsc{};
    /// The stop of the execution of the high-load loop.
    std::atomic<uint64_t> StopTsc{};

    auto operator=(const Metrics& Other) -> Metrics& {
      if (this == &Other) {
        return *this;
      }

      Iterations.store(Other.Iterations.load());
      StartTsc.store(Other.StartTsc.load());
      StopTsc.store(Other.StopTsc.load());
      return *this;
    }
  };

  /// Create the datastructure that is shared between a load worker thread and firestarter.
  /// \arg Id The id of the load worker thread. They are counted from 0 to the maximum number of threads - 1.
  /// \arg OsIndex The os index to which this thread should be bound.
  /// \arg ProcessorInfos The reference to the ProcessorInfos which allows getting the current timestamp.
  /// \arg FunctionPtr The config that is cloned for this specific load worker.
  /// \arg Topology The reference to the processor topology abstraction which allows setting thread affinity.
  /// \arg LoadVar The variable that controls the execution of the load worker.
  /// \arg Period Is used in combination with the LoadVar for the low load routine.
  /// \arg DumpRegisters Should the code to support dumping registers be baked into the high load routine of the
  /// compiled payload.
  /// \arg ErrorDetection Should the code to support error detection between thread be baked into the high load routine
  /// of the compiled payload.
  LoadWorkerData(uint64_t Id, uint64_t OsIndex, std::shared_ptr<ProcessorInformation> ProcessorInfos,
                 const std::unique_ptr<platform::PlatformConfig>& FunctionPtr, const CPUTopology& Topology,
                 volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period, bool DumpRegisters,
                 bool ErrorDetection)
      : LoadVar(LoadVar)
      , Period(Period)
      , DumpRegisters(DumpRegisters)
      , ErrorDetection(ErrorDetection)
      , Id(Id)
      , OsIndex(OsIndex)
      , ProcessorInfos(std::move(ProcessorInfos))
      , Topology(Topology)
      , Config(FunctionPtr->clone()) {}

  ~LoadWorkerData() = default;

  /// Set the shared pointer to the memory shared between two thread for the communication required for the error
  /// detection feature.
  /// \arg CommunicationLeft The memory shared with the left thread.
  /// \arg CommunicationRight The memory shared with the right thread.
  void setErrorCommunication(std::shared_ptr<uint64_t> CommunicationLeft,
                             std::shared_ptr<uint64_t> CommunicationRight) {
    this->CommunicationLeft = std::move(CommunicationLeft);
    this->CommunicationRight = std::move(CommunicationRight);
  }

  /// Gettter for the id of the thread.
  [[nodiscard]] auto id() const -> uint64_t { return Id; }
  /// Getter for the current platform config.
  [[nodiscard]] auto config() const -> const platform::PlatformConfig& { return *Config; }

  /// Access the DumpRegisterStruct. Asserts when dumping registers is not enabled.
  /// \returns a reference to the DumpRegisterStruct
  [[nodiscard]] auto dumpRegisterStruct() const -> DumpRegisterStruct& {
    assert(DumpRegisters && "Tried to access DumpRegisterStruct, but dumping registers is not enabled.");
    return Memory->ExtraVars.Drs;
  }

  /// Access the ErrorDetectionStruct. Asserts when error detections is not enabled.
  /// \returns a reference to the ErrorDetectionStruct
  [[nodiscard]] auto errorDetectionStruct() const -> ErrorDetectionStruct& {
    assert(ErrorDetection && "Tried to access ErrorDetectionStruct, but error detection is not enabled.");
    return Memory->ExtraVars.Eds;
  }

  /// The members in this struct are used for the communication between the main thread and the load thread.
  struct Communication {
    /// The state of the load worker.
    LoadThreadState State = LoadThreadState::ThreadWait;
    /// This variable will be set to true when the state change was acknowledged by the load thread.
    bool Ack = false;
    /// The mutex that is used to lock access to the Ack and State variabels.
    std::mutex Mutex;
  } Communication;

  /// The memory which is used by the load worker.
  LoadWorkerMemory::UniquePtr Memory = {nullptr, nullptr};

  /// The compiled payload which contains the pointers to the specific functions which are executed and some stats.
  payload::CompiledPayload::UniquePtr CompiledPayloadPtr = {nullptr, nullptr};

  /// The variable that controls the execution of the load worker.
  volatile LoadThreadWorkType& LoadVar;

  /// The size of the buffer that is allocated in the load worker.
  uint64_t BuffersizeMem{};

  /// The collected metrics from the current execution of the LoadThreadState::ThreadWork state. Do not read from it.
  Metrics CurrentRun;

  /// The collected metrics from the last execution of the LoadThreadState::ThreadWork state.
  Metrics LastRun;

  // period in usecs
  // used in low load routine to sleep 1/100th of this time
  std::chrono::microseconds Period;

  /// Should the code to support dumping registers be baked into the high load routine of the compiled payload.
  bool DumpRegisters;

  /// Should the code to support error detection between thread be baked into the high load routine of the compiled
  /// payload.
  bool ErrorDetection;
  /// The pointer to the variable that is used for communication to the left thread for the error detection feature.
  std::shared_ptr<uint64_t> CommunicationLeft;
  /// The pointer to the variable that is used for communication to the right thread for the error detection feature.
  std::shared_ptr<uint64_t> CommunicationRight;

  /// The id of this load thread.
  const uint64_t Id;
  /// The os index to which this thread should be bound.
  const uint64_t OsIndex;
  /// The reference to the environment which allows getting the current timestamp.
  std::shared_ptr<ProcessorInformation> ProcessorInfos;
  /// The reference to the processor topology abstraction which allows setting thread affinity.
  const CPUTopology& Topology;
  /// The config that is cloned from the environment for this specfic load worker.
  std::unique_ptr<platform::PlatformConfig> Config;
};

} // namespace firestarter
