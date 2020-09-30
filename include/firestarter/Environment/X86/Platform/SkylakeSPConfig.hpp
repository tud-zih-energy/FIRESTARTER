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

#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKESPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKESPCONFIG_H

#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class SkylakeSPConfig : public X86PlatformConfig {

public:
  SkylakeSPConfig(const asmjit::x86::Features *supportedFeatures,
                  unsigned family, unsigned model, unsigned threads)
      : X86PlatformConfig("SKL_XEONEP", 6, {85}, {1, 2}, 0,
                          {32768, 1048576, 1441792}, 1048576000, family, model,
                          threads,
                          new payload::AVX512Payload(supportedFeatures)){};

  ~SkylakeSPConfig(){};

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_S", 3},
                                                          {"RAM_P", 1},
                                                          {"L3_S", 1},
                                                          {"L3_P", 1},
                                                          {"L2_S", 4},
                                                          {"L2_L", 70},
                                                          {"L1_S", 0},
                                                          {"L1_L", 40},
                                                          {"L1_BROADCAST", 120},
                                                          {"REG", 160}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
