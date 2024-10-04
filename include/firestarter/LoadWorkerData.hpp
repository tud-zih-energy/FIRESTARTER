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
#include "DumpRegisterStruct.hpp"
#include "Environment/Environment.hpp"
#include "ErrorDetectionStruct.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

#define PAD_SIZE(size, align) align*(int)std::ceil((double)size / (double)align)

#if defined(__APPLE__)
#define ALIGNED_MALLOC(size, align) aligned_alloc(align, PAD_SIZE(size, align))
#define ALIGNED_FREE free
#elif defined(__MINGW64__)
#define ALIGNED_MALLOC(size, align) _mm_malloc(PAD_SIZE(size, align), align)
#define ALIGNED_FREE _mm_free
#elif defined(_MSC_VER)
#define ALIGNED_MALLOC(size, align) _aligned_malloc(PAD_SIZE(size, align), align)
#define ALIGNED_FREE _aligned_free
#else
#define ALIGNED_MALLOC(size, align) std::aligned_alloc(align, PAD_SIZE(size, align))
#define ALIGNED_FREE std::free
#endif

namespace firestarter {

/// This struct holds the data for optional FIRESTARTER functionalities.
struct ExtraLoadWorkerVariables {
  /// The data for the dump registers functionality.
  DumpRegisterStruct Drs;
  /// The data for the error detections functionality.
  ErrorDetectionStruct Eds;
};

/// This struct is used to allocate the memory for the high-load routine.
struct LoadWorkerMemory {
  /// The extra variables that are before the memory used for the calculation in the high-load routine. They are used
  /// for features where further communication between the high-load routine is needed e.g., for error detection or
  /// dumping registers.
  ExtraLoadWorkerVariables ExtraVars;

  /// A placeholder to extract the address of the memory region with dynamic size which is used for the calculation in
  /// the high-load routine. Do not write or read to this type directly.
  EightBytesType DoNotUseAddrMem;

  /// This padding makes shure that we are aligned to a cache line. The allocated memory will most probably reach beyond
  /// this array.
  EightBytesType DoNotUsePadding[7];

public:
  /// Get the pointer to the start of the memory use for computations.
  /// \returns the pointer to the memory.
  [[nodiscard]] auto getMemoryAddress() -> auto{ return reinterpret_cast<double*>(&DoNotUseAddrMem); }

  /// Get the offset to the memory which is used by the high-load functions
  /// \returns the offset to the memory
  [[nodiscard]] constexpr static auto getMemoryOffset() -> auto{ return offsetof(LoadWorkerMemory, DoNotUseAddrMem); }
};

class LoadWorkerData {
public:
  LoadWorkerData(int Id, environment::Environment& Environment, volatile LoadThreadWorkType& LoadVar, uint64_t Period,
                 bool DumpRegisters, bool ErrorDetection)
      : LoadVar(LoadVar)
      , Period(Period)
      , DumpRegisters(DumpRegisters)
      , ErrorDetection(ErrorDetection)
      , Id(Id)
      , Environment(Environment)
      , Config(new environment::platform::RuntimeConfig(Environment.selectedConfig())) {}

  ~LoadWorkerData() {
    delete Config;
    if (Memory != nullptr) {
      ALIGNED_FREE(Memory);
    }
  }

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

  LoadThreadState State = LoadThreadState::ThreadWait;
  bool Ack = false;
  std::mutex Mutex;

  LoadWorkerMemory* Memory = nullptr;

  volatile LoadThreadWorkType& LoadVar;
  uint64_t BuffersizeMem{};
  uint64_t Iterations = 0;
  // save the last iteration count when switching payloads
  std::atomic<uint64_t> LastIterations{};
  uint64_t Flops{};
  uint64_t StartTsc{};
  uint64_t StopTsc{};
  std::atomic<uint64_t> LastStartTsc{};
  std::atomic<uint64_t> LastStopTsc{};
  // period in usecs
  // used in low load routine to sleep 1/100th of this time
  uint64_t Period;
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
