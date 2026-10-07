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
/// For AArch64 we use a generic model since there is no direct equivalent
/// to x86 family/model/stepping.
class AArch64CpuModel : public CpuModel {
private:
  /// A generic identifier for the AArch64 model
  unsigned ModelId;

public:
  AArch64CpuModel() = delete;
  explicit AArch64CpuModel(unsigned ModelId) noexcept : ModelId(ModelId) {}

  /// \arg Other The model to which operator < should be checked.
  /// \return true if this is less than other
  [[nodiscard]] auto operator<(const CpuModel& Other) const -> bool override {
    const auto* DerivedModel = dynamic_cast<const AArch64CpuModel*>(&Other);
    if (!DerivedModel) {
      throw std::runtime_error("Other is not of the correct type AArch64CpuModel");
    }

    return ModelId < DerivedModel->ModelId;
  }

  /// Check if two models match.
  /// \arg Other The model to which equality should be checked.
  /// \return true if this and the other model match
  [[nodiscard]] auto operator==(const CpuModel& Other) const -> bool override {
    const auto* DerivedModel = dynamic_cast<const AArch64CpuModel*>(&Other);
    if (!DerivedModel) {
      throw std::runtime_error("Other is not of the correct type AArch64CpuModel");
    }

    return ModelId == DerivedModel->ModelId;
  }
};

} // namespace firestarter::aarch64
