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

#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

namespace firestarter {

auto MetricName::fromString(const std::string& Metric) -> MetricName {
  // Group 1 matches the minus sign, group 2 the root metric name, group 4 the sub-metric name
  const std::regex Re(R"(^(-)?([\w-]+)(/([\w-]+))?$)");
  std::smatch M;

  if (std::regex_match(Metric, M, Re)) {
    const bool Inverted = M[1].length() == 1;
    auto RootMetricName = M[2].str();
    std::optional<std::string> SubmetricName;
    if (M[4].matched) {
      SubmetricName = M[4].str();
    }
    return {Inverted, RootMetricName, SubmetricName};
  }

  throw std::invalid_argument("Could not parse the metric name: " + Metric);
}

} // namespace firestarter