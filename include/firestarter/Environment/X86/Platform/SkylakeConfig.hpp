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

#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKECONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKECONFIG_H

#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class SkylakeConfig final : public X86PlatformConfig {

public:
  SkylakeConfig(asmjit::CpuFeatures const &supportedFeatures, unsigned family,
                unsigned model, unsigned threads)
      : X86PlatformConfig("SKL_COREI", 6, {78, 94}, {1, 2}, 0,
                          {32768, 262144, 1572864}, 104857600, 1536, family,
                          model, threads,
                          new payload::FMAPayload(supportedFeatures)) {}

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings() const override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_L", 3},
                                                          {"L3_LS_256", 5},
                                                          {"L2_LS_256", 18},
                                                          {"L1_2LS_256", 78},
                                                          {"REG", 40}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
