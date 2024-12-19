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

#include "firestarter/X86/X86CpuModel.hpp"

#include <gtest/gtest.h>

namespace {

class InvalidCpuModel : public firestarter::CpuModel {
  [[nodiscard]] auto operator<(const CpuModel& /*Other*/) const -> bool override { return true; }

  [[nodiscard]] auto operator==(const CpuModel& /*Other*/) const -> bool override { return true; }
};

} // namespace

TEST(X86CpuModel, X86CpuModelAllowed) {
  firestarter::x86::X86CpuModel Model(/*FamilyId=*/1, /*ModelId=*/2);

  EXPECT_NO_THROW((void)(Model == Model));
  EXPECT_ANY_THROW((void)(Model == InvalidCpuModel()));
}

TEST(X86CpuModel, CheckEqual) {
  firestarter::x86::X86CpuModel Model(/*FamilyId=*/1, /*ModelId=*/2);

  EXPECT_TRUE(Model == Model);
  EXPECT_FALSE(Model == firestarter::x86::X86CpuModel(/*FamilyId=*/1, /*ModelId=*/0));
  EXPECT_FALSE(Model == firestarter::x86::X86CpuModel(/*FamilyId=*/0, /*ModelId=*/2));
  EXPECT_FALSE(Model == firestarter::x86::X86CpuModel(/*FamilyId=*/3, /*ModelId=*/4));
}

TEST(X86CpuModel, CheckLess) {
  firestarter::x86::X86CpuModel Model(/*FamilyId=*/1, /*ModelId=*/2);

  EXPECT_TRUE(firestarter::x86::X86CpuModel(/*FamilyId=*/0, /*ModelId=*/1) < Model);
  EXPECT_TRUE(firestarter::x86::X86CpuModel(/*FamilyId=*/0, /*ModelId=*/2) < Model);
  EXPECT_TRUE(firestarter::x86::X86CpuModel(/*FamilyId=*/0, /*ModelId=*/3) < Model);

  EXPECT_TRUE(firestarter::x86::X86CpuModel(/*FamilyId=*/1, /*ModelId=*/1) < Model);
  EXPECT_FALSE(Model < Model);
  EXPECT_FALSE(firestarter::x86::X86CpuModel(/*FamilyId=*/1, /*ModelId=*/3) < Model);

  EXPECT_FALSE(firestarter::x86::X86CpuModel(/*FamilyId=*/2, /*ModelId=*/1) < Model);
  EXPECT_FALSE(firestarter::x86::X86CpuModel(/*FamilyId=*/2, /*ModelId=*/2) < Model);
  EXPECT_FALSE(firestarter::x86::X86CpuModel(/*FamilyId=*/2, /*ModelId=*/3) < Model);
}