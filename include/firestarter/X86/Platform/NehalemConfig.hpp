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

#include "firestarter/X86/Payload/SSE2Payload.hpp"
#include "firestarter/X86/Platform/X86PlatformConfig.hpp"

namespace firestarter::x86::platform {
class NehalemConfig final : public X86PlatformConfig {
public:
  NehalemConfig() noexcept
      : X86PlatformConfig(
            /*Name=*/"NHM_COREI", /*Family=*/6, /*Models=*/{30, 37, 23},
            /*Settings=*/
            firestarter::payload::PayloadSettings(/*Threads=*/{1, 2}, /*DataCacheBufferSize=*/{32768, 262144, 1572864},
                                                  /*RamBufferSize=*/104857600, /*Lines=*/1536,
                                                  /*Groups=*/
                                                  InstructionGroups{{{"RAM_P", 1}, {"L1_LS", 70}, {"REG", 2}}}),
            /*Payload=*/std::make_shared<const payload::SSE2Payload>()) {}
};
} // namespace firestarter::x86::platform
