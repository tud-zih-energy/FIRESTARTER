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

#include "firestarter/X86/X86CpuFeatures.hpp"

#include <asmjit/asmjit.h>
#include <gtest/gtest.h>

namespace {

class InvalidCpuFeatures : public firestarter::CpuFeatures {
  [[nodiscard]] auto hasAll(const CpuFeatures& /*Features*/) const -> bool override { return true; };
};

} // namespace

TEST(X86CpuFeatures, X86CpuFeaturesAllowed) {
  firestarter::x86::X86CpuFeatures Features{};

  EXPECT_NO_THROW((void)Features.hasAll(Features));
  EXPECT_ANY_THROW((void)Features.hasAll(InvalidCpuFeatures()));
}

TEST(X86CpuFeatures, CheckHasAll) {
  const auto Features =
      firestarter::x86::X86CpuFeatures().add(asmjit::CpuFeatures::X86::kAVX2).add(asmjit::CpuFeatures::X86::kAVX512_F);

  EXPECT_FALSE(Features.hasAll(firestarter::x86::X86CpuFeatures().add(asmjit::CpuFeatures::X86::kAVX)));
  EXPECT_TRUE(Features.hasAll(firestarter::x86::X86CpuFeatures().add(asmjit::CpuFeatures::X86::kAVX2)));
  EXPECT_TRUE(Features.hasAll(firestarter::x86::X86CpuFeatures().add(asmjit::CpuFeatures::X86::kAVX512_F)));
  EXPECT_TRUE(Features.hasAll(Features));
}