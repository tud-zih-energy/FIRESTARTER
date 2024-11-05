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

#include "CompiledPayload.hpp"
#include "firestarter/Constants.hpp"
#include "firestarter/Environment/CPUTopology.hpp"
#include "firestarter/Environment/Payload/PayloadSettings.hpp"
#include <chrono>
#include <list>
#include <string>
#include <utility>
#include <vector>

namespace firestarter::environment::payload {

class Payload {
private:
  /// The name of this payload. It is usally named by the CPU extension this payload uses e.g., SSE2 or FMA.
  std::string Name;

  /// The size of the SIMD registers in units of doubles (8B)
  unsigned RegisterSize = 0;

  /// The number of SIMD registers used by the payload
  unsigned RegisterCount = 0;

  /// Get the number of items in the sequence that start with a given string.
  /// \arg Sequence The sequence that is analyzed.
  /// \arg Start The string that contains the start of the item names that should be counted in the sequence.
  /// \returns The number of items in the sequence that start with the supplied strings.
  [[nodiscard]] static auto getSequenceStartCount(const std::vector<std::string>& Sequence, const std::string& Start)
      -> unsigned;

protected:
  /// Generate a sequence of items interleaved with one another based on a supplied number how many times each items
  /// should appear in the resulting sequence.
  /// \arg Proportion The mapping of items defined by a string and the number of times this item should apear in the
  /// resuling sequence.
  /// \returns The sequence that is generated from the supplied propotions
  [[nodiscard]] static auto generateSequence(const std::vector<std::pair<std::string, unsigned>>& Proportion)
      -> std::vector<std::string>;

  /// Get the number of items in the sequence that start with "L2".
  /// \arg Sequence The sequence that is analyzed.
  /// \returns The number of items items in the sequence that start with "L2".
  [[nodiscard]] static auto getL2SequenceCount(const std::vector<std::string>& Sequence) -> unsigned {
    return getSequenceStartCount(Sequence, "L2");
  };

  /// Get the number of items in the sequence that start with "L3".
  /// \arg Sequence The sequence that is analyzed.
  /// \returns The number of items items in the sequence that start with "L3".
  [[nodiscard]] static auto getL3SequenceCount(const std::vector<std::string>& Sequence) -> unsigned {
    return getSequenceStartCount(Sequence, "L3");
  };

  /// Get the number of items in the sequence that start with "RAM".
  /// \arg Sequence The sequence that is analyzed.
  /// \returns The number of items items in the sequence that start with "RAM".
  [[nodiscard]] static auto getRAMSequenceCount(const std::vector<std::string>& Sequence) -> unsigned {
    return getSequenceStartCount(Sequence, "RAM");
  };

  /// Get the maximum number of repetitions of the the supplied sequence so that the size of the sequence times the
  /// number of repetitions is smaller equal to the number of lines. The number of repetitions is a unsigned number.
  /// \arg Sequence The reference to the sequence that should be repeated multiple times
  /// \arg NumberOfLines The maximum number of entries in the repeated sequence
  /// \returns The number of repetitions of the sequence.
  [[nodiscard]] static auto getNumberOfSequenceRepetitions(const std::vector<std::string>& Sequence,
                                                           const unsigned NumberOfLines) -> unsigned {
    if (Sequence.empty()) {
      return 0;
    }
    return NumberOfLines / Sequence.size();
  };

  /// Get the number of accesses that can be made to 80% of the L2 cache size (each incrementing the pointer to the
  /// cache) before the pointer need to be reseted to the original value. This assumes that each L2 item in the sequence
  /// increments the pointer by one cache line (64B). It is also assumed that the number of accesses fit at least once
  /// into this cache. This should always be the case on modern CPUs.
  /// \arg Sequence The reference to the sequence.
  /// \arg NumberOfLines The maximum number of entries in the repeated sequence.
  /// \arg Size The size of the L2 Cache.
  /// \returns The maximum number of iterations of the repeated sequence to fill up to 80% of the L2 cache.
  [[nodiscard]] static auto getL2LoopCount(const std::vector<std::string>& Sequence, unsigned NumberOfLines,
                                           unsigned Size) -> unsigned;

  /// Get the number of accesses that can be made to 80% of the L3 cache size (each incrementing the pointer to the
  /// cache) before the pointer need to be reseted to the original value. This assumes that each L3 item in the sequence
  /// increments the pointer by one cache line (64B). See the note about assumptions on the size of the cache in the
  /// documentation of getL2LoopCount.
  /// \arg Sequence The reference to the sequence.
  /// \arg NumberOfLines The maximum number of entries in the repeated sequence.
  /// \arg Size The size of the L3 Cache.
  /// \returns The maximum number of iterations of the repeated sequence to fill up to 80% of the L3 cache.
  [[nodiscard]] static auto getL3LoopCount(const std::vector<std::string>& Sequence, unsigned NumberOfLines,
                                           unsigned Size) -> unsigned;

  /// Get the number of accesses that can be made to 100% of the RAM size (each incrementing the pointer to the ram)
  /// before the pointer need to be reseted to the original value. This assumes that each RAM item in the sequence
  /// increments the pointer by one cache line (64B). See the note about assumptions on the size of the cache in the
  /// documentation of getL2LoopCount.
  /// \arg Sequence The reference to the sequence.
  /// \arg NumberOfLines The maximum number of entries in the repeated sequence.
  /// \arg Size The size of the RAM.
  /// \returns The maximum number of iterations of the repeated sequence to fill up to 100% of the RAM.
  [[nodiscard]] static auto getRAMLoopCount(const std::vector<std::string>& Sequence, unsigned NumberOfLines,
                                            unsigned Size) -> unsigned;

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
  [[nodiscard]] virtual auto isAvailable(const CPUTopology& Topology) const -> bool = 0;

  /// Compile this payload with supplied settings and optional features.
  /// \arg Settings The settings for this payload e.g., the number of lines or the size of the caches.
  /// \arg DumpRegisters Should the code to support dumping registers be baked into the high load routine of the
  /// compiled payload.
  /// \arg ErrorDetection Should the code to support error detection between thread be baked into the high load routine
  /// of the compiled payload.
  /// \returns The compiled payload that provides access to the init and load functions.
  [[nodiscard]] virtual auto compilePayload(const PayloadSettings& Settings, bool DumpRegisters,
                                            bool ErrorDetection) const -> CompiledPayload::UniquePtr = 0;

  /// Get the available instruction items that are supported by this payload.
  /// \returns The available instruction items that are supported by this payload.
  [[nodiscard]] virtual auto getAvailableInstructions() const -> std::list<std::string> = 0;
};

} // namespace firestarter::environment::payload
