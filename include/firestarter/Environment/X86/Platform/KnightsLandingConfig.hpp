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

#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class KnightsLandingConfig : public X86PlatformConfig {

public:
  KnightsLandingConfig(const asmjit::x86::Features *supportedFeatures,
                       unsigned family, unsigned model, unsigned threads)
      : X86PlatformConfig("KNL_XEONPHI", 6, {87}, {4}, 0,
                          {32768, 524288, 236279125}, 26214400, 1536, family,
                          model, threads,
                          new payload::AVX512Payload(supportedFeatures)){};
  ~KnightsLandingConfig(){};

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>(
        {{"RAM_P", 3}, {"L2_S", 8}, {"L1_L", 40}, {"REG", 10}});
  }
};
} // namespace firestarter::environment::x86::platform
