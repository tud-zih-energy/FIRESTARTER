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

#include "firestarter/Environment/X86/Payload/FMAPayload.hpp"
#include "firestarter/Environment/X86/Platform/X86PlatformConfig.hpp"

namespace firestarter::environment::x86::platform {
class RomeConfig final : public X86PlatformConfig {
public:
  RomeConfig() noexcept
      : X86PlatformConfig(
            /*Name=*/"ZEN_2_EPYC", /*Family=*/23, /*Models=*/{49},
            /*Settings=*/
            environment::payload::PayloadSettings(
                /*Threads=*/{1, 2}, /*DataCacheBufferSize=*/{32768, 524288, 2097152}, /*RamBufferSize=*/104857600,
                /*Lines=*/1536,
                /*InstructionGroups=*/
                {{"RAM_L", 10}, {"L3_L", 25}, {"L2_L", 91}, {"L1_2LS_256", 72}, {"L1_LS_256", 82}, {"REG", 75}}),
            /*Payload=*/std::make_shared<const payload::FMAPayload>()) {}
};
} // namespace firestarter::environment::x86::platform
