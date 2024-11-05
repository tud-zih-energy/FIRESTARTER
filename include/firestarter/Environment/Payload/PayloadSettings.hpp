/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
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

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <list>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace firestarter::environment::payload {

struct PayloadSettings {
public:
  using InstructionWithProportion = std::pair<std::string, unsigned>;

private:
  /// The number of threads for which this payload is available. Multiple ones may exsists. The PayloadSettings are
  /// concreate once this is set to contain only one element.
  std::list<unsigned> Threads;
  std::optional<unsigned> InstructionCacheSize;
  std::list<unsigned> DataCacheBufferSize;
  unsigned RamBufferSize;
  unsigned Lines;
  std::vector<InstructionWithProportion> InstructionGroups;

  /// Get the number of items in the sequence that start with a given string.
  /// \arg Sequence The sequence that is analyzed.
  /// \arg Start The string that contains the start of the item names that should be counted in the sequence.
  /// \returns The number of items in the sequence that start with the supplied strings.
  [[nodiscard]] static auto getSequenceStartCount(const std::vector<std::string>& Sequence, const std::string& Start)
      -> unsigned;

public:
  PayloadSettings() = delete;

  PayloadSettings(std::initializer_list<unsigned> Threads, std::initializer_list<unsigned> DataCacheBufferSize,
                  unsigned RamBufferSize, unsigned Lines, std::vector<InstructionWithProportion>&& InstructionGroups)
      : Threads(Threads)
      , DataCacheBufferSize(DataCacheBufferSize)
      , RamBufferSize(RamBufferSize)
      , Lines(Lines)
      , InstructionGroups(std::move(InstructionGroups)) {}

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

  /// Are the payload settings concreate, i.e. can one specific payload be compiled with these settings. This is the
  /// case if the option of threads is reduces to a single element.
  [[nodiscard]] auto isConcreate() const -> bool { return Threads.size() == 1; }

  /// The number of threads which are available with the associated platform/payload.
  [[nodiscard]] auto threads() const -> const auto& { return Threads; }

  /// The concreate number of threads which is selected.
  [[nodiscard]] auto thread() const -> unsigned {
    assert(isConcreate() && "Number of threads is not concreate.");
    return Threads.front();
  }

  /// The available instruction cache size. This refers to the L1i-Cache on the physical CPU core.
  [[nodiscard]] auto instructionCacheSize() const -> const auto& { return InstructionCacheSize; }
  /// The size of the L1d,L2,...,L3 caches per physical CPU core.
  [[nodiscard]] auto dataCacheBufferSize() const -> const auto& { return DataCacheBufferSize; }
  /// The selected size of the buffer that is in the RAM on the physical CPU core.
  [[nodiscard]] auto ramBufferSize() const -> auto{ return RamBufferSize; }
  /// Return the total buffer size for the data caches and the ram per physical CPU core.
  [[nodiscard]] auto totalBufferSize() const -> std::size_t {
    std::size_t Total = 0;
    for (const auto& DataCacheSize : DataCacheBufferSize) {
      Total += DataCacheSize;
    }
    Total += RamBufferSize;
    return Total;
  }
  /// The number of instruction groups which should be used in the payload per physical CPU core.
  [[nodiscard]] auto lines() const -> auto{ return Lines; }

  /// The available instruction cache size. This refers to the L1i-Cache per thread on the physical CPU core.
  [[nodiscard]] auto instructionCacheSizePerThread() const -> std::optional<unsigned> {
    auto InstructionCacheSize = this->InstructionCacheSize;
    if (*InstructionCacheSize) {
      return *InstructionCacheSize / thread();
    }
    return {};
  }
  /// The size of the L1d,L2,...,L3 caches per thread on the physical CPU core.
  [[nodiscard]] auto dataCacheBufferSizePerThread() const -> std::list<unsigned> {
    auto DataCacheBufferSizePerThread = DataCacheBufferSize;
    for (auto& Value : DataCacheBufferSizePerThread) {
      Value /= thread();
    }
    return DataCacheBufferSizePerThread;
  }
  /// The selected size of the buffer that is in the RAM per thread on the physical CPU core.
  [[nodiscard]] auto ramBufferSizePerThread() const -> auto{ return RamBufferSize / thread(); }
  /// Return the total buffer size for the data caches and the ram per thread on the physical CPU core.
  [[nodiscard]] auto totalBufferSizePerThread() const -> std::size_t { return totalBufferSize() / thread(); }
  /// The number of instruction groups which should be used in the payload per thread on the physical CPU core.
  [[nodiscard]] auto linesPerThread() const -> auto{ return Lines / thread(); }

  /// The vector of instruction groups with proportions.
  [[nodiscard]] auto instructionGroups() const -> const auto& { return InstructionGroups; }

  /// Generate a sequence of items interleaved with one another based on the instruction groups.
  /// \returns The sequence that is generated from the supplied propotions in the instruction groups.
  [[nodiscard]] auto sequence() const -> std::vector<std::string> { return generateSequence(instructionGroups()); }

  /// The vector of instructions that are saved in the instruction groups
  [[nodiscard]] auto instructionGroupItems() const -> std::vector<std::string> {
    std::vector<std::string> Items;
    Items.reserve(InstructionGroups.size());
    for (auto const& Pair : InstructionGroups) {
      Items.push_back(Pair.first);
    }
    return Items;
  }

  [[nodiscard]] auto getInstructionGroupsString() const -> std::string {
    std::stringstream Ss;

    for (auto const& [Name, Value] : InstructionGroups) {
      Ss << Name << ":" << Value << ",";
    }

    auto Str = Ss.str();
    if (!Str.empty()) {
      Str.pop_back();
    }

    return Str;
  }

  /// Make the settings concreate.
  /// \arg InstructionCacheSize The detected size of the instructions cache.
  /// \arg ThreadPerCore The number of threads per pysical CPU.
  void concretize(std::optional<unsigned> InstructionCacheSize, unsigned ThreadsPerCore) {
    this->InstructionCacheSize = InstructionCacheSize;
    this->Threads = {ThreadsPerCore};
  }

  /// Save the supplied instruction groups with their proportion in the payload settings.
  /// \arg InstructionGroups The vector with pairs of instructions and proportions
  void selectInstructionGroups(std::vector<InstructionWithProportion> const& InstructionGroups) {
    this->InstructionGroups = InstructionGroups;
  }

  /// Save the line count in the payload settings.
  void setLineCount(unsigned LineCount) { this->Lines = LineCount; }
};

} // namespace firestarter::environment::payload
