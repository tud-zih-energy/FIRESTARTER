/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
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

#include "firestarter/AArch64/Payload/AArch64NEONFMAPayload.hpp"
#include "firestarter/AArch64/Platform/AArch64PlatformConfig.hpp"

namespace firestarter::aarch64::platform {

/// Default AArch64 platform config using NEON FMA payload.
/// Known ARM cores are listed by their MIDR_EL1 fields (Implementer, PartNum, Revision).
class AArch64DefaultConfig final : public AArch64PlatformConfig {
public:
  AArch64DefaultConfig() noexcept
      : AArch64PlatformConfig(
            /*Name=*/"AARCH64_Default", /*RequestedModels=*/{},
            /*Settings=*/
            firestarter::payload::PayloadSettings(
                /*Threads=*/{1, 2, 3}, /*DataCacheBufferSize=*/{16384, 1048576, 786432},
                /*RamBufferSize=*/104857600, /*Lines=*/1536,
                /*Groups=*/
                InstructionGroups{{{"RAM_L", 1}, {"L3_L", 1}, {"L2_L", 5}, {"L1_L", 38}, {"REG", 45}}}),
            /*Payload=*/std::make_shared<const payload::AArch64NEONFMAPayload>()) {}
};

} // namespace firestarter::aarch64::platform
