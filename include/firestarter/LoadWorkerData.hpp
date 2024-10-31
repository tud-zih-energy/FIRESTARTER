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

#include "Constants.hpp"
#include "Environment/Environment.hpp"
#include "LoadWorkerMemory.hpp"
#include "firestarter/Environment/Platform/PlatformConfig.hpp"
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <utility>

namespace firestarter {

class LoadWorkerData {
public:
  struct Metrics {
    std::atomic<uint64_t> Iterations{};
    std::atomic<uint64_t> StartTsc{};
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

  LoadWorkerData(uint64_t Id, const environment::Environment& Environment, volatile LoadThreadWorkType& LoadVar,
                 std::chrono::microseconds Period, bool DumpRegisters, bool ErrorDetection)
      : LoadVar(LoadVar)
      , Period(Period)
      , DumpRegisters(DumpRegisters)
      , ErrorDetection(ErrorDetection)
      , Id(Id)
      , Environment(Environment)
      , Config(Environment.config().clone()) {}

  ~LoadWorkerData() = default;

  void setErrorCommunication(std::shared_ptr<uint64_t> CommunicationLeft,
                             std::shared_ptr<uint64_t> CommunicationRight) {
    this->CommunicationLeft = std::move(CommunicationLeft);
    this->CommunicationRight = std::move(CommunicationRight);
  }

  [[nodiscard]] auto id() const -> uint64_t { return Id; }
  [[nodiscard]] auto environment() const -> const environment::Environment& { return Environment; }
  [[nodiscard]] auto config() const -> environment::platform::PlatformConfig& { return *Config; }

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
  environment::payload::CompiledPayload::UniquePtr CompiledPayloadPtr = {nullptr, nullptr};

  volatile LoadThreadWorkType& LoadVar;
  uint64_t BuffersizeMem{};

  /// The collected metrics from the current execution of the LoadThreadState::ThreadWork state. Do not read from it.
  Metrics CurrentRun;

  /// The collected metrics from the last execution of the LoadThreadState::ThreadWork state.
  Metrics LastRun;

  // period in usecs
  // used in low load routine to sleep 1/100th of this time
  std::chrono::microseconds Period;
  bool DumpRegisters;
  bool ErrorDetection;
  std::shared_ptr<uint64_t> CommunicationLeft;
  std::shared_ptr<uint64_t> CommunicationRight;

  const uint64_t Id;
  const environment::Environment& Environment;
  std::unique_ptr<environment::platform::PlatformConfig> Config;
};

} // namespace firestarter
