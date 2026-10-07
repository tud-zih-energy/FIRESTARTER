/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2026 TU Dresden, Center for Information Services and High
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
class TurinDenseConfig final : public X86PlatformConfig {
public:
  TurinDenseConfig() noexcept
      : X86PlatformConfig(
            /*Name=*/"ZEN_5C_EPYC", /*RequestedModels=*/
            {X86CpuModel(/*FamilyId=*/26, /*ModelId=*/17)},
            /*Settings=*/
            firestarter::payload::PayloadSettings(
                /*Threads=*/{1, 2},
                // 48KiB L1 per core
                // 1MiB L2 per core
                // 2MiB L3 per core
                /*DataCacheBufferSize=*/{49152, 1048576, 2097152},
                /*RamBufferSize=*/104857600, /*Lines=*/1536,
                /*Groups=*/
                InstructionGroups{{{"REG", 36},
                                   {"L1_L", 78},
                                   {"L1_2L", 5},
                                   {"L1_BROADCAST", 94},
                                   {"L1_LS", 93},
                                   {"L2_L", 3},
                                   {"L2_LS", 28},
                                   {"L3_L", 80},
                                   {"L3_P", 23},
                                   {"RAM_L", 4}}}),
            /*Payload=*/std::make_shared<const payload::AVX512Payload>()) {}
};
} // namespace firestarter::x86::platform
