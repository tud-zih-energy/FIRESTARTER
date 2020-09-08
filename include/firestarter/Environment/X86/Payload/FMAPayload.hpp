/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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

#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class FMAPayload : public X86Payload {
public:
  FMAPayload(const asmjit::x86::Features *const supportedFeatures)
      : X86Payload(supportedFeatures,
                   {asmjit::x86::Features::kAVX, asmjit::x86::Features::kFMA},
                   "FMA"){};

  int compilePayload(std::vector<std::pair<std::string, unsigned>> proportion,
                     std::list<unsigned> dataCacheBufferSize,
                     unsigned ramBufferSize, unsigned thread,
                     unsigned numberOfLines) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;

  firestarter::environment::payload::Payload *clone(void) override {
    return new FMAPayload(this->supportedFeatures);
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

#endif
