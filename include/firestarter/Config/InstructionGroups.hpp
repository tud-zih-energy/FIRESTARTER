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

#include <sstream>
#include <string>
#include <vector>

namespace firestarter {

/// Struct to parse selected from a string. The format is a comma delimited list of instruction value pairs. The values
/// are unsigned integers.
struct InstructionGroups {
  InstructionGroups() = default;

  friend auto operator<<(std::ostream& Stream, const InstructionGroups& IGroups) -> std::ostream&;

  /// Parse the instruction group string. It is a comma delimited list of instruction value pairs. The values are
  /// unsigned integers.
  /// \arg Groups The instruction groups as a string.
  [[nodiscard]] static auto fromString(const std::string& Groups) -> InstructionGroups;

  /// The vector of used instructions that are saved in the instruction groups
  [[nodiscard]] auto intructions() const -> std::vector<std::string> {
    std::vector<std::string> Items;
    Items.reserve(Groups.size());
    for (auto const& Pair : Groups) {
      Items.push_back(Pair.first);
    }
    return Items;
  }

  /// The parsed instruction groups
  std::vector<std::pair<std::string, unsigned>> Groups;
};

inline auto operator<<(std::ostream& Stream, const InstructionGroups& IGroups) -> std::ostream& {
  std::stringstream Ss;

  for (auto const& [Key, Value] : IGroups.Groups) {
    Ss << Key << ":" << Value << ",";
  }

  auto S = Ss.str();
  if (!S.empty()) {
    S.pop_back();
  }

  Stream << S;
  return Stream;
}

} // namespace firestarter