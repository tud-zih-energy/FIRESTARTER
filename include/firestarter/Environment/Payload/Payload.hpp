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
#include "firestarter/Environment/CPUTopology.hpp"
#include "firestarter/Environment/Payload/PayloadStats.hpp"
#include <chrono>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace firestarter::environment::payload {

class Payload;

class CompiledPayload {
public:
  CompiledPayload() = delete;
  virtual ~CompiledPayload() = default;

  using UniquePtr = std::unique_ptr<CompiledPayload, void (*)(CompiledPayload*)>;

  using HighLoadFunctionPtr = uint64_t (*)(double*, volatile LoadThreadWorkType*, uint64_t);

  CompiledPayload(const PayloadStats& Stats, std::unique_ptr<Payload>&& PayloadPtr,
                  HighLoadFunctionPtr HighLoadFunction)
      : Stats(Stats)
      , PayloadPtr(std::move(PayloadPtr))
      , HighLoadFunction(HighLoadFunction) {}

  [[nodiscard]] auto stats() const -> const PayloadStats& { return Stats; };

  void init(double* MemoryAddr, uint64_t BufferSize);

  void lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period);

  [[nodiscard]] auto highLoadFunction(double* AddrMem, volatile LoadThreadWorkType& LoadVar, uint64_t Iterations)
      -> uint64_t {
    return HighLoadFunction(AddrMem, &LoadVar, Iterations);
  }

protected:
  // We need to access this pointer directly to free the associated memory from asmjit
  [[nodiscard]] auto highLoadFunctionPtr() -> HighLoadFunctionPtr { return HighLoadFunction; }

private:
  PayloadStats Stats;

  std::unique_ptr<Payload> PayloadPtr;

  HighLoadFunctionPtr HighLoadFunction;
};

class Payload {
private:
  std::string Name;
  [[nodiscard]] static auto getSequenceStartCount(const std::vector<std::string>& Sequence, const std::string& Start)
      -> unsigned;

  /// The size of the SIMD registers in units of doubles (8B)
  unsigned RegisterSize = 0;
  /// The number of SIMD registers used by the payload
  unsigned RegisterCount = 0;

protected:
  [[nodiscard]] static auto generateSequence(const std::vector<std::pair<std::string, unsigned>>& Proportion)
      -> std::vector<std::string>;
  [[nodiscard]] static auto getL2SequenceCount(const std::vector<std::string>& Sequence) -> unsigned {
    return getSequenceStartCount(Sequence, "L2");
  };
  [[nodiscard]] static auto getL3SequenceCount(const std::vector<std::string>& Sequence) -> unsigned {
    return getSequenceStartCount(Sequence, "L3");
  };
  [[nodiscard]] static auto getRAMSequenceCount(const std::vector<std::string>& Sequence) -> unsigned {
    return getSequenceStartCount(Sequence, "RAM");
  };

  [[nodiscard]] static auto getNumberOfSequenceRepetitions(const std::vector<std::string>& Sequence,
                                                           const unsigned NumberOfLines) -> unsigned {
    if (Sequence.empty()) {
      return 0;
    }
    return NumberOfLines / Sequence.size();
  };

  [[nodiscard]] static auto getL2LoopCount(const std::vector<std::string>& Sequence, unsigned NumberOfLines,
                                           unsigned Size, unsigned Threads) -> unsigned;
  [[nodiscard]] static auto getL3LoopCount(const std::vector<std::string>& Sequence, unsigned NumberOfLines,
                                           unsigned Size, unsigned Threads) -> unsigned;
  [[nodiscard]] static auto getRAMLoopCount(const std::vector<std::string>& Sequence, unsigned NumberOfLines,
                                            unsigned Size, unsigned Threads) -> unsigned;

  virtual void init(double* MemoryAddr, uint64_t BufferSize) const = 0;

  virtual void lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) const = 0;

public:
  Payload() = delete;

  Payload(std::string Name, unsigned RegisterSize, unsigned RegisterCount)
      : Name(std::move(Name))
      , RegisterSize(RegisterSize)
      , RegisterCount(RegisterCount) {}
  virtual ~Payload() = default;

  friend void CompiledPayload::init(double*, uint64_t);
  friend void CompiledPayload::lowLoadFunction(volatile LoadThreadWorkType&, std::chrono::microseconds);

  [[nodiscard]] auto name() const -> const std::string& { return Name; }
  /// The size of the SIMD registers in units of doubles (8B)
  [[nodiscard]] auto registerSize() const -> unsigned { return RegisterSize; }
  /// The number of SIMD registers used by the payload
  [[nodiscard]] auto registerCount() const -> unsigned { return RegisterCount; }

  [[nodiscard]] virtual auto isAvailable(const CPUTopology*) const -> bool = 0;

  [[nodiscard]] virtual auto compilePayload(std::vector<std::pair<std::string, unsigned>> const& Proportion,
                                            unsigned InstructionCacheSize,
                                            std::list<unsigned> const& DataCacheBufferSize, unsigned RamBufferSize,
                                            unsigned Thread, unsigned NumberOfLines, bool DumpRegisters,
                                            bool ErrorDetection) const -> CompiledPayload::UniquePtr = 0;
  [[nodiscard]] virtual auto getAvailableInstructions() const -> std::list<std::string> = 0;

  [[nodiscard]] virtual auto clone() const -> std::unique_ptr<Payload> = 0;
};

} // namespace firestarter::environment::payload
