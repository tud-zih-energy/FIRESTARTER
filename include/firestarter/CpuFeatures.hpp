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

/// Abstract class that defines the methods required to check if cpu features are available.
class CpuFeatures {
public:
  CpuFeatures() = default;
  virtual ~CpuFeatures() = default;

  /// Check if this class has all features which are given in the argument.
  /// \arg Features The features which should be check if they are available.
  /// \returns true if this class has all features given in the argument.
  [[nodiscard]] virtual auto hasAll(const CpuFeatures& Features) const -> bool = 0;
};

} // namespace firestarter
