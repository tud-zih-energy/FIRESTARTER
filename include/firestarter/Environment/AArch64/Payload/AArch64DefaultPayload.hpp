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

#include <firestarter/Environment/AArch64/Payload/AArch64Payload.hpp>

namespace firestarter::environment::aarch64::payload {

class AArch64DefaultPayload final : public AArch64Payload {
public:
  AArch64DefaultPayload(asmjit::CpuFeatures const &supportedFeatures)
      : AArch64Payload(
            supportedFeatures,
            {asmjit::CpuFeatures::ARM::kARMv8a},
            "ARMv8a", 2, 32) {}

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
    return new AArch64DefaultPayload(this->supportedFeatures());
  };

private:
  const std::map<std::string, unsigned> instructionFlops = {
      {"REG", 8}, {"L1_L", 0}, {"L1_S", 0}, {"L2_L", 0}, {"L2_S", 0},
      {"L3_L", 0}, {"L3_2", 0}, {"RAM_L", 0}, {"RAM_S", 0}
  };

  const std::map<std::string, unsigned> instructionMemory = {
      {"RAM_L", 64}, {"RAM_S", 128}, {"RAM_LS", 128}, {"RAM_P", 64}
  };
};
} // namespace firestarter::environment::aarch64::payload
