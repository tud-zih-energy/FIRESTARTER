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

#include "firestarter/Constants.hpp"
#include "firestarter/LoadWorkerMemory.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/Payload/Payload.hpp"
#include "firestarter/AArch64/AArch64CpuFeatures.hpp"

#include <asmjit/a64.h>
#include <cassert>
#include <cstdint>
#include <map>
#include <type_traits>
#include <utility>

constexpr const auto AArch64InitBlocksize = 1024;

namespace firestarter::aarch64::payload {

/// This abstract class models a payload that can be compiled with settings and executed for AArch64 CPUs.
class AArch64Payload : public firestarter::payload::Payload {
private:
  /// This list contains the features (cpu extensions) that are required to execute the payload.
  AArch64CpuFeatures FeatureRequests;

  /// The mapping from instructions to the number of flops per instruction.
  std::map<std::string, unsigned> InstructionFlops;

  /// The mapping from instructions to the size of main memory accesses for this instruction.
  std::map<std::string, unsigned> InstructionMemory;

public:
  /// Abstract constructor for a payload on AArch64 CPUs.
  AArch64Payload(AArch64CpuFeatures FeatureRequests, std::string Name, unsigned RegisterSize, unsigned RegisterCount,
                 std::map<std::string, unsigned>&& InstructionFlops,
                 std::map<std::string, unsigned>&& InstructionMemory) noexcept
      : Payload(std::move(Name), RegisterSize, RegisterCount)
      , FeatureRequests(std::move(FeatureRequests))
      , InstructionFlops(std::move(InstructionFlops))
      , InstructionMemory(std::move(InstructionMemory)) {}

  /// Check if this payload is available on the current system.
  [[nodiscard]] auto isAvailable(const CpuFeatures& Features) const -> bool final {
    return Features.hasAll(FeatureRequests);
  }

  /// The features that are required for this payload
  [[nodiscard]] auto featureRequests() const -> const auto& { return FeatureRequests; }

  /// The mapping from instructions to the number of flops per instruction.
  [[nodiscard]] auto instructionFlops() const -> const auto& { return InstructionFlops; }

  /// The mapping from instructions to the size of main memory accesses.
  [[nodiscard]] auto instructionMemory() const -> const auto& { return InstructionMemory; }

protected:
  /// Print the generated assembler Code of asmjit
  static void printAssembler(asmjit::BaseBuilder& Builder) {
    asmjit::String Sb;
    asmjit::Formatter::formatNodeList(Sb, asmjit::FormatOptions{}, &Builder);
    log::info() << Sb.data();
  }

  /// Initialize the memory used by the high load function.
  static void initMemory(double* MemoryAddr, uint64_t BufferSize, double FirstValue, double LastValue);

  /// Function to produce a low load on the cpu.
  void lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) const final;

public:
  /// Get the available instruction items that are supported by this payload.
  [[nodiscard]] auto getAvailableInstructions() const -> std::list<std::string>;
};

} // namespace firestarter::aarch64::payload
