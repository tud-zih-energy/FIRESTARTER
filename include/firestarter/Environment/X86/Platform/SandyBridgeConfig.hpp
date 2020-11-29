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

#pragma once

#include <firestarter/Environment/X86/Payload/AVXPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class SandyBridgeConfig : public X86PlatformConfig {

public:
  SandyBridgeConfig(const asmjit::x86::Features *supportedFeatures,
                    unsigned family, unsigned model, unsigned threads)
      : X86PlatformConfig("SNB_COREI", 6, {42, 58}, {1, 2}, 0,
                          {32768, 262144, 1572864}, 104857600, 1536, family,
                          model, threads,
                          new payload::AVXPayload(supportedFeatures)){};
  ~SandyBridgeConfig(){};

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_L", 2},
                                                          {"L3_LS", 4},
                                                          {"L2_LS", 10},
                                                          {"L1_LS", 90},
                                                          {"REG", 45}});
  }
};
} // namespace firestarter::environment::x86::platform
