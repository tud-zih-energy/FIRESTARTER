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

#include "firestarter/X86/Platform/X86PlatformConfig.hpp"
#include "firestarter/X86/Payload/X86Payload.hpp"

#include <gtest/gtest.h>

namespace {

class X86PayloadTest : public firestarter::x86::payload::X86Payload {
public:
  X86PayloadTest()
      : firestarter::x86::payload::X86Payload(
            /*FeatureRequests=*/firestarter::x86::X86CpuFeatures(),
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

class X86PlafromConfigTest : public firestarter::x86::platform::X86PlatformConfig {
public:
  inline static const auto Model1 = firestarter::x86::X86CpuModel(/*FamilyId=*/1, /*ModelId=*/2);
  inline static const auto Model2 = firestarter::x86::X86CpuModel(/*FamilyId=*/3, /*ModelId=*/4);
  inline static const auto InvalidModel = firestarter::x86::X86CpuModel(/*FamilyId=*/5, /*ModelId=*/6);

  X86PlafromConfigTest()
      : firestarter::x86::platform::X86PlatformConfig(
            "X86PlatformConfig", {Model1, Model2},
            firestarter::payload::PayloadSettings(
                /*Threads=*/{}, /*DataCacheBufferSize=*/{}, /*RamBufferSize=*/0,
                /*Lines=*/0, /*Groups=*/firestarter::InstructionGroups(firestarter::InstructionGroups::InternalType())),
            std::make_shared<X86PayloadTest>()) {}
};

class TrueCpuFeatures : public firestarter::CpuFeatures {
  [[nodiscard]] auto hasAll(const CpuFeatures& /*Features*/) const -> bool override { return true; };
};

class FalseCpuFeatures : public firestarter::CpuFeatures {
  [[nodiscard]] auto hasAll(const CpuFeatures& /*Features*/) const -> bool override { return false; };
};

} // namespace

TEST(X86PlafromConfigTest, CheckIsDefault) {
  EXPECT_TRUE(X86PlafromConfigTest().isDefault(X86PlafromConfigTest::Model1, TrueCpuFeatures()));
  EXPECT_FALSE(X86PlafromConfigTest().isDefault(X86PlafromConfigTest::Model1, FalseCpuFeatures()));

  EXPECT_TRUE(X86PlafromConfigTest().isDefault(X86PlafromConfigTest::Model2, TrueCpuFeatures()));
  EXPECT_FALSE(X86PlafromConfigTest().isDefault(X86PlafromConfigTest::Model2, FalseCpuFeatures()));

  EXPECT_FALSE(X86PlafromConfigTest().isDefault(X86PlafromConfigTest::InvalidModel, TrueCpuFeatures()));
  EXPECT_FALSE(X86PlafromConfigTest().isDefault(X86PlafromConfigTest::InvalidModel, FalseCpuFeatures()));
}