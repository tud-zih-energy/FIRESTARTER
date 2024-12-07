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

#include <string>
#include <vector>

namespace firestarter {

/// Struct to parse the CPU binding list from a string. The format is "x,y,z", "x-y", "x-y/step", and any combination.
struct CpuBind {
  /// Parse the cpu bind string and return a vector containing the parsed number of all selected cpus.
  /// \arg CpuBindString The string containing the cpu binding in the format "x,y,z", "x-y", "x-y/step", and any
  /// combination.
  /// \returns The vector of parsed CPUs.
  static auto fromString(const std::string& CpuBindString) -> std::vector<uint64_t>;
};

} // namespace firestarter