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

#include "../Payload/SSE2Payload.hpp"
#include "X86PlatformConfig.hpp"

namespace firestarter::environment::x86::platform {
class NehalemEPConfig final : public X86PlatformConfig {
public:
  NehalemEPConfig(asmjit::CpuFeatures const& SupportedFeatures, unsigned Family, unsigned Model)
      : X86PlatformConfig("NHM_XEONEP", 6, {26, 44}, {1, 2}, 0, {32768, 262144, 2097152}, 104857600, 1536, Family,
                          Model, std::make_unique<payload::SSE2Payload>(SupportedFeatures)) {}

  [[nodiscard]] auto getDefaultPayloadSettings() const -> std::vector<std::pair<std::string, unsigned>> override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_P", 1}, {"L1_LS", 60}, {"REG", 2}});
  }
};
} // namespace firestarter::environment::x86::platform
