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

namespace firestarter {

/// Abstract class that defines the methods required to check if one cpu model is equal to another
class CpuModel {
public:
  CpuModel() = default;
  virtual ~CpuModel() = default;

  /// \arg Other The model to which operator < should be checked.
  /// \return true if this is less than other
  [[nodiscard]] virtual auto operator<(const CpuModel& Other) const -> bool = 0;

  /// Check if two models match.
  /// \arg Other The model to which equality should be checked.
  /// \return true if this and the other model match
  [[nodiscard]] virtual auto operator==(const CpuModel& Other) const -> bool = 0;
};

} // namespace firestarter
