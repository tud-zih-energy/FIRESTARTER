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
            /*Name=*/"AARCH64_Default", /*RequestedModels=*/
            {
                // ARM Cortex-A53 (Implementer: 0x41, Part: 0x000)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x000, /*Revision=*/0),
                // ARM Cortex-A55 (Implementer: 0x41, Part: 0x001)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x001, /*Revision=*/0),
                // ARM Cortex-A57 (Implementer: 0x41, Part: 0x00F)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x00F, /*Revision=*/0),
                // ARM Cortex-A72 (Implementer: 0x41, Part: 0x00D)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x00D, /*Revision=*/0),
                // ARM Cortex-A73 (Implementer: 0x41, Part: 0x00E)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x00E, /*Revision=*/0),
                // ARM Cortex-A75 (Implementer: 0x41, Part: 0x00C)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x00C, /*Revision=*/0),
                // ARM Cortex-A76 (Implementer: 0x41, Part: 0x013)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x013, /*Revision=*/0),
                // ARM Cortex-A77 (Implementer: 0x41, Part: 0x015)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x015, /*Revision=*/0),
                // ARM Cortex-A78 (Implementer: 0x41, Part: 0x014)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x014, /*Revision=*/0),
                // ARM Cortex-A710 (Implementer: 0x41, Part: 0x017)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x017, /*Revision=*/0),
                // ARM Cortex-A715 (Implementer: 0x41, Part: 0x018)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x018, /*Revision=*/0),
                // ARM Cortex-X1 (Implementer: 0x41, Part: 0x01E)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x01E, /*Revision=*/0),
                // ARM Cortex-X2 (Implementer: 0x41, Part: 0x021)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x021, /*Revision=*/0),
                // ARM Cortex-X3 (Implementer: 0x41, Part: 0x023)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x023, /*Revision=*/0),
                // ARM Cortex-X4 (Implementer: 0x41, Part: 0x024)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x024, /*Revision=*/0),
                // ARM Neoverse N1 (Implementer: 0x41, Part: 0x001)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x001, /*Revision=*/1),
                // ARM Neoverse N2 (Implementer: 0x41, Part: 0x022)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x022, /*Revision=*/0),
                // ARM Neoverse V1 (Implementer: 0x41, Part: 0x002)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x002, /*Revision=*/0),
                // ARM Neoverse V2 (Implementer: 0x41, Part: 0x020)
                AArch64CpuModel(/*Implementer=*/0x41, /*PartNum=*/0x020, /*Revision=*/0),
            },
            /*Settings=*/
            firestarter::payload::PayloadSettings(
                /*Threads=*/{1, 2, 3}, /*DataCacheBufferSize=*/{16384, 1048576, 786432},
                /*RamBufferSize=*/104857600, /*Lines=*/1536,
                /*Groups=*/
                InstructionGroups{{{"RAM_L", 1}, {"L3_L", 1}, {"L2_L", 5}, {"L1_L", 38}, {"REG", 45}}}),
            /*Payload=*/std::make_shared<const payload::AArch64NEONFMAPayload>()) {}
};

} // namespace firestarter::aarch64::platform
