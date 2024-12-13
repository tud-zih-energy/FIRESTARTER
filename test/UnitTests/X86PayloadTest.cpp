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

#include "firestarter/X86/Payload/X86Payload.hpp"

#include <asmjit/asmjit.h>
#include <gtest/gtest.h>

class X86PayloadTest : public firestarter::x86::payload::X86Payload {
public:
  X86PayloadTest()
      : firestarter::x86::payload::X86Payload(
            /*FeatureRequests=*/firestarter::x86::X86CpuFeatures()
                .add(asmjit::CpuFeatures::X86::kAVX2)
                .add(asmjit::CpuFeatures::X86::kAVX512_F),
            /*Name=*/"X86Payload", /*RegisterSize=*/0,
            /*RegisterCount=*/0,
            /*InstructionFlops=*/
            {},
            /*InstructionMemory=*/{}) {}

  void init(double* /*MemoryAddr*/, uint64_t /*BufferSize*/) const override {}

  [[nodiscard]] auto compilePayload(const firestarter::payload::PayloadSettings& /*Settings*/, bool /*DumpRegisters*/,
                                    bool /*ErrorDetection*/, bool /*PrintAssembler*/) const
      -> firestarter::payload::CompiledPayload::UniquePtr override {
    return {nullptr, nullptr};
  }
};

class TrueCpuFeatures : public firestarter::CpuFeatures {
  [[nodiscard]] auto hasAll(const CpuFeatures& /*Features*/) const -> bool override { return true; };
};

class FalseCpuFeatures : public firestarter::CpuFeatures {
  [[nodiscard]] auto hasAll(const CpuFeatures& /*Features*/) const -> bool override { return false; };
};

TEST(X86PayloadTest, CpuFeatureHasAllReturned) {
  EXPECT_TRUE(X86PayloadTest().isAvailable(TrueCpuFeatures()));
  EXPECT_FALSE(X86PayloadTest().isAvailable(FalseCpuFeatures()));
}

TEST(X86CpuFeatures, CheckIsAvailable) {
  EXPECT_FALSE(X86PayloadTest().isAvailable(firestarter::x86::X86CpuFeatures().add(asmjit::CpuFeatures::X86::kAVX)));
  EXPECT_FALSE(X86PayloadTest().isAvailable(firestarter::x86::X86CpuFeatures().add(asmjit::CpuFeatures::X86::kAVX2)));
  EXPECT_FALSE(
      X86PayloadTest().isAvailable(firestarter::x86::X86CpuFeatures().add(asmjit::CpuFeatures::X86::kAVX512_F)));
  EXPECT_TRUE(X86PayloadTest().isAvailable(X86PayloadTest().featureRequests()));
  EXPECT_TRUE(X86PayloadTest().isAvailable(firestarter::x86::X86CpuFeatures()
                                               .add(asmjit::CpuFeatures::X86::kAVX)
                                               .add(asmjit::CpuFeatures::X86::kAVX2)
                                               .add(asmjit::CpuFeatures::X86::kAVX512_F)));
}