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

#include "firestarter/AArch64/Payload/AArch64Payload.hpp"
#include "firestarter/Payload/PayloadSettings.hpp"

namespace firestarter::aarch64::payload {

/// This payload is designed for ARMv8a NEON with FMA instructions.
class AArch64NEONFMAPayload final : public AArch64Payload {
public:
  AArch64NEONFMAPayload() noexcept
      : AArch64Payload(
            /*FeatureRequests=*/AArch64CpuFeatures().add(asmjit::CpuFeatures::ARM::kARMv8a),
            /*Name=*/"ARMv8a_NEON_FMA", /*RegisterSize=*/2,
            /*RegisterCount=*/32,
            /*InstructionFlops=*/
            {{"REG", 16},
             {"L1_L", 0},
             {"L1_S", 0},
             {"L2_L", 0},
             {"L2_S", 0},
             {"L3_L", 0},
             {"L3_S", 0},
             {"RAM_L", 0},
             {"RAM_S", 0}},
            /*InstructionMemory=*/{{"RAM_L", 64}, {"RAM_S", 128}, {"RAM_LS", 128}, {"RAM_P", 64}}) {}

  [[nodiscard]] auto compilePayload(const firestarter::payload::PayloadSettings& Settings, bool DumpRegisters,
                                    bool ErrorDetection, bool PrintAssembler,
                                    firestarter::payload::HighLoadControlFlowDescription ControlFlow) const
      -> firestarter::payload::CompiledPayload::UniquePtr override;

  [[nodiscard]] auto getAvailableInstructions() const -> std::list<std::string> override;

private:
  /// Function to initialize the memory used by the high load function.
  void init(double* MemoryAddr, uint64_t BufferSize) const override;
};

} // namespace firestarter::aarch64::payload
