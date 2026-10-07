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

#include "firestarter/AArch64/Platform/AArch64DefaultConfig.hpp"
#include "firestarter/FunctionSelection.hpp"

#include <memory>

namespace firestarter::aarch64 {

/// Function selection for AArch64 platforms.
class AArch64FunctionSelection final : public FunctionSelection {
public:
  AArch64FunctionSelection() = default;

  [[nodiscard]] auto platformConfigs() const
      -> const std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>>& override {
    return PlatformConfigs;
  }

  [[nodiscard]] auto fallbackPlatformConfigs() const
      -> const std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>>& override {
    return FallbackPlatformConfigs;
  }

private:
  /// The list of available platform configs.
  std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>> PlatformConfigs = {
      std::make_shared<platform::AArch64DefaultConfig>()};

  /// The list of fallback configs.
  std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>> FallbackPlatformConfigs = {
      std::make_shared<platform::AArch64DefaultConfig>()};
};

} // namespace firestarter::aarch64
