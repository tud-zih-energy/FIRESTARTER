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

#include <firestarter/Environment/X86/Payload/ZENFMAPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class NaplesConfig : public X86PlatformConfig {

public:
  NaplesConfig(asmjit::x86::Features const &supportedFeatures, unsigned family,
               unsigned model, unsigned threads)
      : X86PlatformConfig("ZEN_EPYC", 23, {1, 8, 17, 24}, {1, 2}, 0,
                          {65536, 524288, 2097152}, 104857600, 1536, family,
                          model, threads,
                          new payload::ZENFMAPayload(supportedFeatures)){};
  ~NaplesConfig(){};

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings() override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_L", 3},
                                                          {"L3_L", 14},
                                                          {"L2_L", 75},
                                                          {"L1_LS", 81},
                                                          {"REG", 100}});
  }
};
} // namespace firestarter::environment::x86::platform
