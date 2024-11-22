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

#include "asmjit/core/cpuinfo.h"
#include "firestarter/Environment/X86/Payload/AVX512Payload.hpp"

namespace firestarter::environment::x86::payload {

/// This payload is designed for the AVX512 foundation CPU extension specialized for AMX.
class AVX512WithAMXPayload : public AVX512Payload {
public:
  AVX512WithAMXPayload() noexcept {
    // Enable the AMX instruction in the AVX512 Payload and request AMX_TILE and AMX_BF16 feature.
    addInstructionFlops("AMX", 512);
    addFeatureRequest(asmjit::CpuFeatures::X86::kAMX_TILE);
    addFeatureRequest(asmjit::CpuFeatures::X86::kAMX_BF16);
  }
};
} // namespace firestarter::environment::x86::payload
