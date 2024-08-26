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

#include <firestarter/Environment/AArch64/Payload/AArch64DefaultPayload.hpp>
#include <firestarter/Environment/AArch64/Platform/AArch64PlatformConfig.hpp>

namespace firestarter::environment::aarch64::platform {
class AArch64DefaultConfig final : public AArch64PlatformConfig {

public:
  AArch64DefaultConfig(asmjit::CpuFeatures const &supportedFeatures,
                  unsigned family, unsigned model, unsigned threads)
      : AArch64PlatformConfig("AARCH64_Default", 21, {1, 2, 3}, {1}, 0,
                          {16384, 1048576, 786432}, 104857600, 1536, family,
                          model, threads,
                          new payload::AArch64DefaultPayload(supportedFeatures)) {}

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings() const override {
    return std::vector<std::pair<std::string, unsigned>>(
        {{"RAM_L", 1}, {"L3_L", 1}, {"L2_L", 5}, {"L1_L", 38}, {"REG", 45}});
  }
};
} // namespace firestarter::environment::aarch64::platform