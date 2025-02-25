/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2025 TU Dresden, Center for Information Services and High
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

#include "firestarter/Config/MetricName.hpp"

#include <regex>

namespace firestarter {

auto MetricName::fromString(const std::string& Metric) -> MetricName {
  const std::regex Re("^(-)(\\w+)(/\\w+)$");
  std::smatch M;

  if (std::regex_match(Metric, M, Re)) {
    bool Inverted = M[0].matched;
    auto RootMetricName = M[1].str();
    std::optional<std::string> SubmetricName;
    if (M[2].matched) {
      SubmetricName = M[2].str();
    }
    return {Inverted, RootMetricName, SubmetricName};
  }

  throw std::invalid_argument("Could not parse the metric name");
}

} // namespace firestarter