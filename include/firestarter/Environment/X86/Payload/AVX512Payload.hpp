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
class AVX512Payload final : public X86Payload {
public:
  AVX512Payload()
      : X86Payload({asmjit::CpuFeatures::X86::kAVX512_F}, "AVX512", 8, 32) {}

  [[nodiscard]] auto compilePayload(std::vector<std::pair<std::string, unsigned>> const& Proportion,
                                    unsigned InstructionCacheSize, std::list<unsigned> const& DataCacheBufferSize,
                                    unsigned RamBufferSize, unsigned Thread, unsigned NumberOfLines, bool DumpRegisters,
                                    bool ErrorDetection) const
      -> environment::payload::CompiledPayload::UniquePtr override;

  [[nodiscard]] auto getAvailableInstructions() const -> std::list<std::string> override;

  [[nodiscard]] auto clone() const -> std::unique_ptr<firestarter::environment::payload::Payload> override {
    return std::make_unique<AVX512Payload>();
  };

private:
  void init(double* MemoryAddr, uint64_t BufferSize) const override;

  const std::map<std::string, unsigned> InstructionFlops = {
      {"REG", 32},   {"L1_L", 32},  {"L1_BROADCAST", 16}, {"L1_S", 16}, {"L1_LS", 16}, {"L2_L", 32},
      {"L2_S", 16},  {"L2_LS", 16}, {"L3_L", 32},         {"L3_S", 16}, {"L3_LS", 16}, {"L3_P", 16},
      {"RAM_L", 32}, {"RAM_S", 16}, {"RAM_LS", 16},       {"RAM_P", 16}};

  const std::map<std::string, unsigned> InstructionMemory = {
      {"RAM_L", 64}, {"RAM_S", 128}, {"RAM_LS", 128}, {"RAM_P", 64}};
};
} // namespace firestarter::environment::x86::payload
