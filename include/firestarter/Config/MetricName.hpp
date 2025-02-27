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

#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace firestarter {

/// This struct handels the name of a metric. It can be created from a string and converted back into a string.
/// The metric name constains the name of the root metric and optionally the name of a submetric. These values are used
/// to check if two metric names are the same. In addition to that the name can contain additional attribute. Currently
/// this is only the flag to invert the metric value for the optimization.
struct MetricName {
  MetricName() = delete;

  MetricName(bool Inverted, std::string RootMetricName, std::optional<std::string> SubmetricName = {})
      : Inverted(Inverted)
      , RootMetricName(std::move(RootMetricName))
      , SubmetricName(std::move(SubmetricName)) {}

  /// Parse the metric name. It consists of an optional minus ('-') the root metric name, and an optional slash ('/')
  /// followed by the submetric name.
  /// \arg MetricName The name of the metric
  [[nodiscard]] static auto fromString(const std::string& Metric) -> MetricName;

  /// Convert the internal metric name to a string following the format: optional minus ('-') for an inverted metric,
  /// the root metric name, and an optional slash ('/') followed by the submetric name.
  [[nodiscard]] auto toString() const -> std::string {
    return (Inverted ? "-" : "") + RootMetricName + (SubmetricName ? ("/" + *SubmetricName) : "");
  }

  /// Convert the metric name to a string that does not contain any information about the attributes.
  [[nodiscard]] auto toStringWithoutAttributes() const -> std::string {
    return RootMetricName + (SubmetricName ? ("/" + *SubmetricName) : "");
  }

  /// Is the metric inverted
  [[nodiscard]] auto inverted() const -> const auto& { return Inverted; }
  /// The name of the root metric
  [[nodiscard]] auto rootMetricName() const -> const auto& { return RootMetricName; }
  /// The optional name of the submetric
  [[nodiscard]] auto submetricName() const -> const auto& { return SubmetricName; }

  auto operator==(const MetricName& Other) const -> bool {
    return std::tie(RootMetricName, SubmetricName) == std::tie(Other.rootMetricName(), Other.submetricName());
  }

private:
  /// True if the metric is inverted. This is an additional attribute that is not user in the comparison of metrics.
  bool Inverted;
  /// The name of the root metric
  std::string RootMetricName;
  /// The optional name of the submetric
  std::optional<std::string> SubmetricName;
};

} // namespace firestarter

template <> struct std::hash<firestarter::MetricName> {
  auto operator()(firestarter::MetricName const& Name) const noexcept -> std::size_t {
    return std::hash<std::string>{}(Name.toStringWithoutAttributes());
  }
};

template <> struct std::less<firestarter::MetricName> {
  auto operator()(firestarter::MetricName const& Lhs, firestarter::MetricName const& Rhs) const noexcept -> bool {
    return std::less<std::string>{}(Lhs.toStringWithoutAttributes(), Rhs.toStringWithoutAttributes());
  }
};