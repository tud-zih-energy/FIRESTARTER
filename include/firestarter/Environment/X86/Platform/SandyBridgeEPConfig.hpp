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

#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGEEPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGEEPCONFIG_H

#include "firestarter/Environment/X86/Payload/AVXPayload.hpp"
#include "firestarter/Environment/X86/Platform/X86PlatformConfig.hpp"

namespace firestarter::environment::x86::platform {
class SandyBridgeEPConfig final : public X86PlatformConfig {
public:
  SandyBridgeEPConfig() noexcept
      : X86PlatformConfig(
            /*Name=*/"SNB_XEONEP", /*Family=*/6, /*Models=*/{45, 62},
            /*Settings=*/
            environment::payload::PayloadSettings(
                /*Threads=*/{1, 2}, /*DataCacheBufferSize=*/{32768, 262144, 2621440}, /*RamBufferSize=*/104857600,
                /*Lines=*/1536,
                /*InstructionGroups=*/{{"RAM_L", 3}, {"L3_LS", 2}, {"L2_LS", 10}, {"L1_LS", 90}, {"REG", 30}}),
            /*Payload=*/std::make_shared<const payload::AVXPayload>()) {}
};
} // namespace firestarter::environment::x86::platform

#endif
