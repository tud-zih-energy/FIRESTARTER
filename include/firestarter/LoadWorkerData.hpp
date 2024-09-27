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

class LoadWorkerData {
public:
  LoadWorkerData(int Id, environment::Environment& Environment, volatile uint64_t* LoadVar, uint64_t Period,
                 bool DumpRegisters, bool ErrorDetection)
      : AddrHigh(LoadVar)
      , Period(Period)
      , DumpRegisters(DumpRegisters)
      , ErrorDetection(ErrorDetection)
      , Id(Id)
      , Environment(Environment)
      , Config(new environment::platform::RuntimeConfig(Environment.selectedConfig())) {
    // use REGISTER_MAX_NUM cache lines for the dumped registers
    // and another cache line for the control variable.
    // as we are doing aligned moves we only have the option to waste a
    // whole cacheline
    AddrOffset += DumpRegisters ? sizeof(DumpRegisterStruct) / sizeof(uint64_t) : 0;

    AddrOffset += ErrorDetection ? sizeof(ErrorDetectionStruct) / sizeof(uint64_t) : 0;
  }

  ~LoadWorkerData() {
    delete Config;
    if (AddrMem - AddrOffset != nullptr) {
      ALIGNED_FREE(AddrMem - AddrOffset);
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

  [[nodiscard]] auto errorDetectionStruct() const -> const ErrorDetectionStruct* {
    return reinterpret_cast<ErrorDetectionStruct*>(AddrMem - AddrOffset);
  }

  int Comm = THREAD_WAIT;
  bool Ack = false;
  std::mutex Mutex;
  uint64_t* AddrMem = nullptr;
  uint64_t AddrOffset = 0;
  volatile uint64_t* AddrHigh;
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
