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
#include <utility>
#include <vector>

namespace firestarter {

/// Struct to parse selected from a string. The format is a comma delimited list of instruction value pairs. The values
/// are unsigned integers.
struct InstructionGroups {
  using InternalType = std::vector<std::pair<std::string, unsigned>>;

  InstructionGroups() = default;

  explicit InstructionGroups(InternalType Groups)
      : Groups(std::move(Groups)) {}

  explicit operator const InternalType&() const noexcept { return Groups; }

  friend auto operator<<(std::ostream& Stream, const InstructionGroups& IGroups) -> std::ostream&;

  /// Parse the instruction group string. It is a comma delimited list of instruction value pairs. The values are
  /// unsigned integers.
  /// \arg Groups The instruction groups as a string.
  [[nodiscard]] static auto fromString(const std::string& Groups) -> InstructionGroups;

  /// Combine instructions and values for these instructions into the combined instruction groups.
  /// \arg Instructions The vector of instructions
  /// \arg Values The vector of values
  /// \returns The combined instruction groups
  [[nodiscard]] static auto fromInstructionAndValues(const std::vector<std::string>& Instructions,
                                                     const std::vector<unsigned>& Values) -> InstructionGroups;

  /// The vector of used instructions that are saved in the instruction groups
  [[nodiscard]] auto intructions() const -> std::vector<std::string>;

private:
  /// The parsed instruction groups
  std::vector<std::pair<std::string, unsigned>> Groups;
};

inline auto operator<<(std::ostream& Stream, const InstructionGroups& Groups) -> std::ostream& {
  std::stringstream Ss;

  for (auto const& [Key, Value] : static_cast<InstructionGroups::InternalType>(Groups)) {
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