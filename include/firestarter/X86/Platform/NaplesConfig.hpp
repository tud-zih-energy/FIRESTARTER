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

#include "firestarter/X86/Payload/ZENFMAPayload.hpp"
#include "firestarter/X86/Platform/X86PlatformConfig.hpp"

namespace firestarter::x86::platform {
class NaplesConfig final : public X86PlatformConfig {
public:
  NaplesConfig() noexcept
      : X86PlatformConfig(
            /*Name=*/"ZEN_EPYC", /*RequestedModels=*/
            {X86CpuModel(/*FamilyId=*/23, /*ModelId=*/1), X86CpuModel(/*FamilyId=*/23, /*ModelId=*/8),
             X86CpuModel(/*FamilyId=*/23, /*ModelId=*/17), X86CpuModel(/*FamilyId=*/23, /*ModelId=*/24)},
            /*Settings=*/
            firestarter::payload::PayloadSettings(
                /*Threads=*/{1, 2}, /*DataCacheBufferSize=*/{65536, 524288, 2097152}, /*RamBufferSize=*/104857600,
                /*Lines=*/1536,
                /*Groups=*/
                InstructionGroups{{{"RAM_L", 3}, {"L3_L", 14}, {"L2_L", 75}, {"L1_LS", 81}, {"REG", 100}}}),
            /*Payload=*/std::make_shared<const payload::ZENFMAPayload>()) {}
};
} // namespace firestarter::x86::platform
