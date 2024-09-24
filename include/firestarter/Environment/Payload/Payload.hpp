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

#include <list>
#include <string>
#include <utility>
#include <vector>

namespace firestarter::environment::payload {

class Payload {
private:
  std::string Name;
  [[nodiscard]] static auto getSequenceStartCount(const std::vector<std::string>& Sequence, const std::string& Start)
      -> unsigned;

protected:
  unsigned Flops = 0;
  unsigned Bytes = 0;
  // number of instructions in load loop
  unsigned Instructions = 0;
  // size of used simd registers in bytes
  unsigned RegisterSize = 0;
  // number of used simd registers
  unsigned RegisterCount = 0;

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
    if (Sequence.size() == 0) {
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

public:
  Payload() = delete;

  Payload(std::string Name, unsigned RegisterSize, unsigned RegisterCount)
      : Name(std::move(Name))
      , RegisterSize(RegisterSize)
      , RegisterCount(RegisterCount) {}
  virtual ~Payload() = default;

  [[nodiscard]] auto name() const -> const std::string& { return Name; }
  [[nodiscard]] auto flops() const -> unsigned { return Flops; }
  [[nodiscard]] auto bytes() const -> unsigned { return Bytes; }
  [[nodiscard]] auto instructions() const -> unsigned { return Instructions; }
  [[nodiscard]] auto registerSize() const -> unsigned { return RegisterSize; }
  [[nodiscard]] auto registerCount() const -> unsigned { return RegisterCount; }

  [[nodiscard]] virtual auto isAvailable() const -> bool = 0;

  virtual void lowLoadFunction(volatile uint64_t* AddrHigh, uint64_t Period) = 0;

  [[nodiscard]] virtual auto compilePayload(std::vector<std::pair<std::string, unsigned>> const& Proportion,
                                            unsigned InstructionCacheSize,
                                            std::list<unsigned> const& DataCacheBufferSize, unsigned RamBufferSize,
                                            unsigned Thread, unsigned NumberOfLines, bool DumpRegisters,
                                            bool ErrorDetection) -> int = 0;
  [[nodiscard]] virtual auto getAvailableInstructions() const -> std::list<std::string> = 0;
  virtual void init(uint64_t* MemoryAddr, uint64_t BufferSize) = 0;
  [[nodiscard]] virtual auto highLoadFunction(uint64_t* AddrMem, volatile uint64_t* AddrHigh, uint64_t Iterations)
      -> uint64_t = 0;

  [[nodiscard]] virtual auto clone() const -> Payload* = 0;
};

} // namespace firestarter::environment::payload
