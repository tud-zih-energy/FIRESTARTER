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

#include "X86Payload.hpp"

namespace firestarter::environment::x86::payload {
class ZENFMAPayload final : public X86Payload {
public:
  ZENFMAPayload() = delete;

  explicit ZENFMAPayload(asmjit::CpuFeatures const& SupportedFeatures)
      : X86Payload(SupportedFeatures, {asmjit::CpuFeatures::X86::Id::kAVX, asmjit::CpuFeatures::X86::Id::kFMA},
                   "ZENFMA", 4, 16) {}

  auto compilePayload(std::vector<std::pair<std::string, unsigned>> const& Proportion, unsigned InstructionCacheSize,
                      std::list<unsigned> const& DataCacheBufferSize, unsigned RamBufferSize, unsigned Thread,
                      unsigned NumberOfLines, bool DumpRegisters, bool ErrorDetection) -> int override;
  [[nodiscard]] auto getAvailableInstructions() const -> std::list<std::string> override;
  void init(double* MemoryAddr, uint64_t BufferSize) override;

  [[nodiscard]] auto clone() const -> std::unique_ptr<firestarter::environment::payload::Payload> override {
    return std::make_unique<ZENFMAPayload>(this->supportedFeatures());
  };

private:
  const std::map<std::string, unsigned> InstructionFlops = {
      {"REG", 8}, {"L1_LS", 8}, {"L2_L", 8}, {"L3_L", 8}, {"RAM_L", 8}};

  const std::map<std::string, unsigned> InstructionMemory = {{"RAM_L", 64}};
};
} // namespace firestarter::environment::x86::payload
