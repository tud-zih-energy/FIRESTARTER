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

namespace firestarter::x86 {

/// This class models the cpu features on the x86_64 platform.
class X86CpuModel : public CpuModel {
private:
  /// The x86 family id
  unsigned FamilyId;
  /// The x86 model id
  unsigned ModelId;

public:
  X86CpuModel() = delete;
  explicit X86CpuModel(unsigned FamilyId, unsigned ModelId) noexcept
      : FamilyId(FamilyId)
      , ModelId(ModelId) {}

  /// \arg Other The model to which operator < should be checked.
  /// \return true if this is less than other
  [[nodiscard]] auto operator<(const CpuModel& Other) const -> bool override {
    const auto* DerivedModel = dynamic_cast<const X86CpuModel*>(&Other);
    if (!DerivedModel) {
      throw std::runtime_error("Other is not of the correct type X86CpuModel");
    }

    return std::tie(FamilyId, ModelId) < std::tie(DerivedModel->FamilyId, DerivedModel->ModelId);
  }

  /// Check if two models match.
  /// \arg Other The model to which equality should be checked.
  /// \return true if this and the other model match
  [[nodiscard]] auto operator==(const CpuModel& Other) const -> bool override {
    const auto* DerivedModel = dynamic_cast<const X86CpuModel*>(&Other);
    if (!DerivedModel) {
      throw std::runtime_error("Other is not of the correct type X86CpuModel");
    }

    return std::tie(FamilyId, ModelId) == std::tie(DerivedModel->FamilyId, DerivedModel->ModelId);
  }
};

} // namespace firestarter::x86
