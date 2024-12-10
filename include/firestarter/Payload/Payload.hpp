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

#include "firestarter/Constants.hpp"
#include "firestarter/Payload/CompiledPayload.hpp"
#include "firestarter/Payload/PayloadSettings.hpp"
#include "firestarter/ProcessorInformation.hpp"

#include <chrono>
#include <list>
#include <string>
#include <utility>

namespace firestarter::payload {

class Payload {
private:
  /// The name of this payload. It is usally named by the CPU extension this payload uses e.g., SSE2 or FMA.
  std::string Name;

  /// The size of the SIMD registers in units of doubles (8B)
  unsigned RegisterSize = 0;

  /// The number of SIMD registers used by the payload
  unsigned RegisterCount = 0;

protected:
  /// Function to initialize the memory used by the high load function.
  /// \arg MemoryAddr The pointer to the memory.
  /// \arg BufferSize The number of doubles that is allocated in MemoryAddr.
  virtual void init(double* MemoryAddr, uint64_t BufferSize) const = 0;

  /// Function to produce a low load on the cpu.
  /// \arg LoadVar The variable that controls the load. If this variable changes from LoadThreadWorkType::LowLoad to
  /// something else this function will return.
  /// \arg Period The period of the low/high load switching. This function may sleep a fraction of this period.
  virtual void lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) const = 0;

public:
  Payload() = delete;

  /// Abstract construction for the payload.
  /// \arg Name The name of this payload. It is usally named by the CPU extension this payload uses e.g., SSE2 or FMA.
  /// \arg RegisterSize The size of the SIMD registers in units of doubles (8B).
  /// \arg RegisterCount The number of SIMD registers used by the payload.
  Payload(std::string Name, unsigned RegisterSize, unsigned RegisterCount) noexcept
      : Name(std::move(Name))
      , RegisterSize(RegisterSize)
      , RegisterCount(RegisterCount) {}
  virtual ~Payload() = default;

  // Allow init and lowLoadFunction functions to be accessed by the CompiledPayload class.
  friend void CompiledPayload::init(double* MemoryAddr, uint64_t BufferSize);
  friend void CompiledPayload::lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period);

  /// Get the name of this payload. It is usally named by the CPU extension this payload uses e.g., SSE2 or FMA.
  [[nodiscard]] auto name() const -> const std::string& { return Name; }

  /// The size of the SIMD registers in units of doubles (8B)
  [[nodiscard]] auto registerSize() const -> unsigned { return RegisterSize; }

  /// The number of SIMD registers used by the payload
  [[nodiscard]] auto registerCount() const -> unsigned { return RegisterCount; }

  /// Check if this payload is available on the current system. This usally translates if the cpu extensions are
  /// available.
  /// \arg Topology The CPUTopology that is used to check agains if this payload is supported.
  /// \returns true if the payload is supported on the given CPUTopology.
  [[nodiscard]] virtual auto isAvailable(const ProcessorInformation& Topology) const -> bool = 0;

  /// Compile this payload with supplied settings and optional features.
  /// \arg Settings The settings for this payload e.g., the number of lines or the size of the caches.
  /// \arg DumpRegisters Should the code to support dumping registers be baked into the high load routine of the
  /// compiled payload.
  /// \arg ErrorDetection Should the code to support error detection between thread be baked into the high load routine
  /// of the compiled payload.
  /// \arg PrintAssembler Should the generated assembler code be logged.
  /// \returns The compiled payload that provides access to the init and load functions.
  [[nodiscard]] virtual auto compilePayload(const PayloadSettings& Settings, bool DumpRegisters, bool ErrorDetection,
                                            bool PrintAssembler) const -> CompiledPayload::UniquePtr = 0;

  /// Get the available instruction items that are supported by this payload.
  /// \returns The available instruction items that are supported by this payload.
  [[nodiscard]] virtual auto getAvailableInstructions() const -> std::list<std::string> = 0;
};

} // namespace firestarter::payload
