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

#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_BULLDOZERCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_BULLDOZERCONFIG_H

#include <firestarter/Environment/X86/Payload/FMA4Payload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class BulldozerConfig : public X86PlatformConfig {

public:
  BulldozerConfig(const asmjit::x86::Features *supportedFeatures,
                  unsigned family, unsigned model, unsigned threads)
      : X86PlatformConfig("BLD_OPTERON", 21, {1, 2, 3}, {1}, 0,
                          {16384, 1048576, 786432}, 104857600, family, model,
                          threads,
                          new payload::FMA4Payload(supportedFeatures)){};
  ~BulldozerConfig(){};

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>(
        {{"RAM_L", 1}, {"L3_L", 1}, {"L2_LS", 5}, {"L1_L", 90}, {"REG", 45}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
