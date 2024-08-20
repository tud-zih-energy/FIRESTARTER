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

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class FMAPayload final : public X86Payload {
public:
  FMAPayload(asmjit::CpuFeatures const &supportedFeatures)
      : X86Payload(supportedFeatures,
                   {asmjit::CpuFeatures::X86::kAVX, asmjit::CpuFeatures::X86::kFMA},
                   "FMA", 4, 16) {}

  int compilePayload(
      std::vector<std::pair<std::string, unsigned>> const &proportion,
      unsigned instructionCacheSize,
      std::list<unsigned> const &dataCacheBufferSize, unsigned ramBufferSize,
      unsigned thread, unsigned numberOfLines, bool dumpRegisters,
      bool errorDetection) override;
  std::list<std::string> getAvailableInstructions() const override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;

  firestarter::environment::payload::Payload *clone() const override {
    return new FMAPayload(this->supportedFeatures());
  };

private:
  const std::map<std::string, unsigned> instructionFlops = {
      {"REG", 16},  {"L1_L", 16},     {"L1_2L", 16},      {"L1_S", 8},
      {"L1_LS", 8}, {"L1_LS_256", 8}, {"L1_2LS_256", 16}, {"L2_L", 16},
      {"L2_S", 8},  {"L2_LS", 8},     {"L2_LS_256", 8},   {"L2_2LS_256", 16},
      {"L3_L", 16}, {"L3_S", 8},      {"L3_LS", 8},       {"L3_LS_256", 8},
      {"L3_P", 8},  {"RAM_L", 16},    {"RAM_S", 8},       {"RAM_LS", 8},
      {"RAM_P", 8}};

  const std::map<std::string, unsigned> instructionMemory = {
      {"RAM_L", 64}, {"RAM_S", 128}, {"RAM_LS", 128}, {"RAM_P", 64}};
};
} // namespace firestarter::environment::x86::payload
