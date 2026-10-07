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

#include "firestarter/CpuModel.hpp"

#include <cassert>
#include <stdexcept>
#include <tuple>

namespace firestarter::aarch64 {

/// This class models the cpu model on the AArch64 platform.
/// Uses fields from the MIDR_EL1 register (Main ID Register), which is the
/// AArch64 equivalent of x86's CPUID family/model/stepping.
class AArch64CpuModel : public CpuModel {
private:
  /// The implementer code (MIDR_EL1[31:24]), e.g. 0x41=ARM, 0x51=Qualcomm, 0x61=Apple
  unsigned Implementer;
  /// The part number (MIDR_EL1[15:4]), e.g. 0x000=Cortex-A53, 0x00F=Cortex-A57, 0x00D=Cortex-A72
  unsigned PartNum;
  /// The revision (MIDR_EL1[3:0]), e.g. 0=r0, 1=r1, ...
  unsigned Revision;

public:
  AArch64CpuModel() = delete;
  explicit AArch64CpuModel(unsigned Implementer, unsigned PartNum, unsigned Revision) noexcept
      : Implementer(Implementer)
      , PartNum(PartNum)
      , Revision(Revision) {}

  /// \arg Other The model to which operator < should be checked.
  /// \return true if this is less than other
  [[nodiscard]] auto operator<(const CpuModel& Other) const -> bool override {
    const auto* DerivedModel = dynamic_cast<const AArch64CpuModel*>(&Other);
    if (!DerivedModel) {
      throw std::runtime_error("Other is not of the correct type AArch64CpuModel");
    }

    return std::tie(Implementer, PartNum, Revision) < std::tie(DerivedModel->Implementer, DerivedModel->PartNum,
                                                                DerivedModel->Revision);
  }

  /// Check if two models match.
  /// \arg Other The model to which equality should be checked.
  /// \return true if this and the other model match
  [[nodiscard]] auto operator==(const CpuModel& Other) const -> bool override {
    const auto* DerivedModel = dynamic_cast<const AArch64CpuModel*>(&Other);
    if (!DerivedModel) {
      throw std::runtime_error("Other is not of the correct type AArch64CpuModel");
    }

    return std::tie(Implementer, PartNum, Revision) == std::tie(DerivedModel->Implementer, DerivedModel->PartNum,
                                                                DerivedModel->Revision);
  }
};

} // namespace firestarter::aarch64
