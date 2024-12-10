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

#include "firestarter/Platform/PlatformConfig.hpp"

namespace firestarter::platform {

/// This struct is used to iterate over the possible thread configurations available on each platform.
struct PlatformConfigAndThreads {
public:
  PlatformConfigAndThreads() = delete;

  /// The shared pointer to the platform config
  std::shared_ptr<PlatformConfig> Config;
  /// One concreate thread count which is available in the given config
  unsigned ThreadCount;

  /// Get the vector of available configurations for a specific platform config.
  /// \arg Config The reference to the platform config
  /// \returns The vector of available configurations for this platform config.
  [[nodiscard]] static auto fromPlatformConfig(const std::shared_ptr<PlatformConfig>& Config)
      -> std::vector<PlatformConfigAndThreads> {
    std::vector<PlatformConfigAndThreads> Vec;

    for (const auto& Thread : Config->settings().threads()) {
      Vec.emplace_back(PlatformConfigAndThreads{Config, Thread});
    }

    return Vec;
  }

  /// Get the vector of available configurations for a vector of platform configs.
  /// \arg Configs The reference to the vector of platform config
  /// \returns The vector of available configurations for the supplied platform config.
  [[nodiscard]] static auto fromPlatformConfigs(const std::vector<std::shared_ptr<PlatformConfig>>& Configs)
      -> std::vector<PlatformConfigAndThreads> {
    std::vector<PlatformConfigAndThreads> Vec;

    for (const auto& Config : Configs) {
      const auto ConfigAndThreads = fromPlatformConfig(Config);
      Vec.insert(Vec.end(), ConfigAndThreads.cbegin(), ConfigAndThreads.cend());
    }

    return Vec;
  }
};

} // namespace firestarter::platform
