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

#include "firestarter/Config/InstructionGroups.hpp"

#include <cassert>
#include <regex>

namespace firestarter {

auto InstructionGroups::fromString(const std::string& Groups) -> InstructionGroups {
  const auto Delimiter = ',';
  const std::regex Re("^(\\w+):(\\d+)$");

  std::stringstream Ss(Groups);
  std::vector<std::pair<std::string, unsigned>> ParsedGroups;

  while (Ss.good()) {
    std::string Token;
    std::smatch M;
    std::getline(Ss, Token, Delimiter);

    if (std::regex_match(Token, M, Re)) {
      auto Num = std::stoul(M[2].str());
      if (Num == 0) {
        throw std::invalid_argument("instruction-group VAL may not contain number 0"
                                    "\n       --run-instruction-groups format: multiple INST:VAL "
                                    "pairs comma-seperated");
      }
      ParsedGroups.emplace_back(M[1].str(), Num);
    } else {
      throw std::invalid_argument("Invalid symbols in instruction-group: " + Token +
                                  "\n       --run-instruction-groups format: multiple INST:VAL "
                                  "pairs comma-seperated");
    }
  }

  return InstructionGroups{ParsedGroups};
}

auto InstructionGroups::fromInstructionAndValues(const std::vector<std::string>& Instructions,
                                                 const std::vector<unsigned>& Values) -> InstructionGroups {
  assert(Instructions.size() == Values.size());

  std::vector<std::pair<std::string, unsigned>> Groups;

  auto It1 = Instructions.begin();
  auto It2 = Values.begin();
  for (; It1 != Instructions.end(); ++It1, ++It2) {
    Groups.emplace_back(*It1, *It2);
  }

  return InstructionGroups{Groups};
}

auto InstructionGroups::intructions() const -> std::vector<std::string> {
  std::vector<std::string> Items;
  Items.reserve(Groups.size());
  for (auto const& Pair : Groups) {
    Items.push_back(Pair.first);
  }
  return Items;
}

} // namespace firestarter