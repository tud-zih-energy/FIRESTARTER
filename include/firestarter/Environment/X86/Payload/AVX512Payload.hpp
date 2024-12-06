/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
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

#include "firestarter/Environment/X86/Payload/X86Payload.hpp"

namespace firestarter::environment::x86::payload {

/// This payload is designed for the AVX512 foundation CPU extension.
class AVX512Payload final : public X86Payload {
public:
  AVX512Payload() noexcept
      : X86Payload(/*FeatureRequests=*/{asmjit::CpuFeatures::X86::kAVX512_F}, /*Name=*/"AVX512", /*RegisterSize=*/8,
                   /*RegisterCount=*/32,
                   /*InstructionFlops=*/
                   {{"REG", 32},
                    {"L1_L", 32},
                    {"L1_BROADCAST", 16},
                    {"L1_S", 16},
                    {"L1_LS", 16},
                    {"L2_L", 32},
                    {"L2_S", 16},
                    {"L2_LS", 16},
                    {"L3_L", 32},
                    {"L3_S", 16},
                    {"L3_LS", 16},
                    {"L3_P", 16},
                    {"RAM_L", 32},
                    {"RAM_S", 16},
                    {"RAM_LS", 16},
                    {"RAM_P", 16}},
                   /*InstructionMemory=*/{{"RAM_L", 64}, {"RAM_S", 128}, {"RAM_LS", 128}, {"RAM_P", 64}}) {}

  /// Compile this payload with supplied settings and optional features.
  /// \arg Settings The settings for this payload e.g., the number of lines or the size of the caches.
  /// \arg DumpRegisters Should the code to support dumping registers be baked into the high load routine of the
  /// compiled payload.
  /// \arg ErrorDetection Should the code to support error detection between thread be baked into the high load routine
  /// of the compiled payload.
  /// \arg PrintAssembler Should the generated assembler code be logged.
  /// \returns The compiled payload that provides access to the init and load functions.
  [[nodiscard]] auto compilePayload(const environment::payload::PayloadSettings& Settings, bool DumpRegisters,
                                    bool ErrorDetection, bool PrintAssembler) const
      -> environment::payload::CompiledPayload::UniquePtr override;

private:
  /// Function to initialize the memory used by the high load function.
  /// \arg MemoryAddr The pointer to the memory.
  /// \arg BufferSize The number of doubles that is allocated in MemoryAddr.
  void init(double* MemoryAddr, uint64_t BufferSize) const override;
};
} // namespace firestarter::environment::x86::payload
