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

#include "firestarter/Config/CpuBind.hpp"

#include <cstdint>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace firestarter {

auto CpuBind::fromString(const std::string& CpuBindString) -> std::set<uint64_t> {
  std::set<uint64_t> ParsedCpus;

  const auto Delimiter = ',';
  const std::regex Re(R"(^(?:(\d+)(?:-([1-9]\d*)(?:\/([1-9]\d*))?)?)$)");

  std::stringstream Ss(CpuBindString);

  while (Ss.good()) {
    std::string Token;
    std::smatch M;
    std::getline(Ss, Token, Delimiter);

    if (std::regex_match(Token, M, Re)) {
      uint64_t Y = 0;
      uint64_t S = 0;

      auto X = std::stoul(M[1].str());
      if (M[2].matched) {
        Y = std::stoul(M[2].str());
      } else {
        Y = X;
      }
      if (M[3].matched) {
        S = std::stoul(M[3].str());
      } else {
        S = 1;
      }
      if (Y < X) {
        throw std::invalid_argument("y has to be >= x in x-y expressions of CPU list: " + Token);
      }
      for (auto I = X; I <= Y; I += S) {
        ParsedCpus.emplace(I);
      }
    } else {
      throw std::invalid_argument("Invalid symbols in CPU list: " + Token);
    }
  }

  return ParsedCpus;
}

} // namespace firestarter