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

#include "AlignedAlloc.hpp"
#include "Constants.hpp"
#include "DumpRegisterStruct.hpp"
#include "Environment/Environment.hpp"
#include "ErrorDetectionStruct.hpp"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace firestarter {

/// This struct is used to allocate the memory for the high-load routine.
struct LoadWorkerMemory {
private:
  LoadWorkerMemory() = default;
  ~LoadWorkerMemory() = default;

  /// Function to deallocate the memory for this struct to be used with unique_ptr.
  /// \arg Ptr The pointer to the memory
  static void deallocate(void* Ptr) {
    static_cast<LoadWorkerMemory*>(Ptr)->~LoadWorkerMemory();
    AlignedAlloc::free(Ptr);
  }

public:
  using UniquePtr = std::unique_ptr<LoadWorkerMemory, void (*)(void*)>;

  /// The extra variables that are before the memory used for the calculation in the high-load routine. They are used
  /// for optional FIRESTARTER features where further communication between the high-load routine is needed e.g., for
  /// error detection or dumping registers.
  struct ExtraLoadWorkerVariables {
    /// The data for the dump registers functionality.
    DumpRegisterStruct Drs;
    /// The data for the error detections functionality.
    ErrorDetectionStruct Eds;
  } ExtraVars;

  /// A placeholder to extract the address of the memory region with dynamic size which is used for the calculation in
  /// the high-load routine. Do not write or read to this type directly.
  EightBytesType DoNotUseAddrMem;

  /// This padding makes shure that we are aligned to a cache line. The allocated memory will most probably reach beyond
  /// this array.
  std::array<EightBytesType, 7> DoNotUsePadding;

  /// Get the pointer to the start of the memory use for computations.
  /// \returns the pointer to the memory.
  [[nodiscard]] auto getMemoryAddress() -> auto{ return reinterpret_cast<double*>(&DoNotUseAddrMem); }

  /// Get the offset to the memory which is used by the high-load functions
  /// \returns the offset to the memory
  [[nodiscard]] constexpr static auto getMemoryOffset() -> auto{ return offsetof(LoadWorkerMemory, DoNotUseAddrMem); }

  /// Allocate the memory for the high-load thread on 64B cache line boundaries and return a unique_ptr.
  /// \arg Bytes The number of bytes allocated for the array whoose start address is returned by the getMemoryAddress
  /// function.
  /// \returns A unique_ptr to the memory for the high-load thread.
  [[nodiscard]] static auto allocate(const std::size_t Bytes) -> UniquePtr {
    // Allocate the memory for the ExtraLoadWorkerVariables (which are 64B aligned) and the data for the high-load
    // routine which may not be 64B aligned.
    static_assert(sizeof(ExtraLoadWorkerVariables) % 64 == 0,
                  "ExtraLoadWorkerVariables is not a multiple of 64B i.e., multiple cachelines.");
    auto* Ptr = AlignedAlloc::malloc(Bytes + sizeof(ExtraLoadWorkerVariables));
    return {static_cast<LoadWorkerMemory*>(Ptr), deallocate};
  }
};

class LoadWorkerData {
public:
  struct Metrics {
    std::atomic<uint64_t> Iterations{};
    std::atomic<uint64_t> StartTsc{};
    std::atomic<uint64_t> StopTsc{};

    auto operator=(const Metrics& Other) -> Metrics& {
      Iterations.store(Other.Iterations.load());
      StartTsc.store(Other.StartTsc.load());
      StopTsc.store(Other.StopTsc.load());
      return *this;
    }
  };

  LoadWorkerData(int Id, environment::Environment& Environment, volatile LoadThreadWorkType& LoadVar,
                 std::chrono::microseconds Period, bool DumpRegisters, bool ErrorDetection)
      : LoadVar(LoadVar)
      , Period(Period)
      , DumpRegisters(DumpRegisters)
      , ErrorDetection(ErrorDetection)
      , Id(Id)
      , Environment(Environment)
      , Config(new environment::platform::RuntimeConfig(Environment.selectedConfig())) {}

  ~LoadWorkerData() { delete Config; }

  void setErrorCommunication(std::shared_ptr<uint64_t> CommunicationLeft,
                             std::shared_ptr<uint64_t> CommunicationRight) {
    this->CommunicationLeft = std::move(CommunicationLeft);
    this->CommunicationRight = std::move(CommunicationRight);
  }

  [[nodiscard]] auto id() const -> int { return Id; }
  [[nodiscard]] auto environment() const -> environment::Environment& { return Environment; }
  [[nodiscard]] auto config() const -> environment::platform::RuntimeConfig& { return *Config; }

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

private:
  int Id;
  environment::Environment& Environment;
  environment::platform::RuntimeConfig* Config;
};

} // namespace firestarter
