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

#include "firestarter/X86/Payload/AVX512Payload.hpp"
#include "firestarter/X86/Platform/X86PlatformConfig.hpp"

namespace firestarter::x86::platform {
class SapphireRapidsConfig final : public X86PlatformConfig {
public:
  SapphireRapidsConfig() noexcept
      : X86PlatformConfig(
            /*Name=*/"SPR_XEONEP", /*RequestedModels=*/
            {X86CpuModel(/*FamilyId=*/6, /*ModelId=*/143)},
            /*Settings=*/
            firestarter::payload::PayloadSettings(
                /*Threads=*/{1, 2},
                // 48KiB L1 per core
                // 2MiB L2 per core
                // 1.875MiB L3 per core
                /*DataCacheBufferSize=*/{49152, 2097152, 1966080},
                /*RamBufferSize=*/1048576000, /*Lines=*/1536,
                /*Groups=*/
                InstructionGroups{
                    {{"REG", 44}, {"L1_L", 84}, {"L1_2L", 90}, {"L1_LS", 17}, {"L2_L", 39}, {"RAM_P", 10}}}),
            /*Payload=*/std::make_shared<const payload::AVX512Payload>()) {}
};
} // namespace firestarter::x86::platform
