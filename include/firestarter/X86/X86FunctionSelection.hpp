/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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

#include "firestarter/FunctionSelection.hpp"
#include "firestarter/X86/Platform/BulldozerConfig.hpp"
#include "firestarter/X86/Platform/HaswellConfig.hpp"
#include "firestarter/X86/Platform/HaswellEPConfig.hpp"
#include "firestarter/X86/Platform/KnightsLandingConfig.hpp"
#include "firestarter/X86/Platform/NaplesConfig.hpp"
#include "firestarter/X86/Platform/NehalemConfig.hpp"
#include "firestarter/X86/Platform/NehalemEPConfig.hpp"
#include "firestarter/X86/Platform/RomeConfig.hpp"
#include "firestarter/X86/Platform/SandyBridgeConfig.hpp"
#include "firestarter/X86/Platform/SandyBridgeEPConfig.hpp"
#include "firestarter/X86/Platform/SkylakeConfig.hpp"
#include "firestarter/X86/Platform/SkylakeSPConfig.hpp"

#include <memory>

namespace firestarter::x86 {

class X86FunctionSelection final : public FunctionSelection {
public:
  X86FunctionSelection() = default;

  [[nodiscard]] auto
  platformConfigs() const -> const std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>>& override {
    return PlatformConfigs;
  }

  [[nodiscard]] auto fallbackPlatformConfigs() const
      -> const std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>>& override {
    return FallbackPlatformConfigs;
  }

private:
  /// The list of availabe platform configs that is printed when supplying the --avail command line argument. The IDs
  /// for these configs are generated by iterating through this list starting with 1. To maintain stable IDs in
  /// FIRESTARTER new configs should be added to the bottom of the list.
  std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>> PlatformConfigs = {
      std::make_shared<platform::KnightsLandingConfig>(), std::make_shared<platform::SkylakeConfig>(),
      std::make_shared<platform::SkylakeSPConfig>(),      std::make_shared<platform::HaswellConfig>(),
      std::make_shared<platform::HaswellEPConfig>(),      std::make_shared<platform::SandyBridgeConfig>(),
      std::make_shared<platform::SandyBridgeEPConfig>(),  std::make_shared<platform::NehalemConfig>(),
      std::make_shared<platform::NehalemEPConfig>(),      std::make_shared<platform::BulldozerConfig>(),
      std::make_shared<platform::NaplesConfig>(),         std::make_shared<platform::RomeConfig>()};

  /// The list of configs that are fallbacks. If none of the PlatformConfigs is the default one on the current CPU, we
  /// select the first one from this list that is available on the current system. If multiple configs can be available
  /// on one system the one with higher priority should be at the top of this list. Modern X86 CPUs will support SSE2
  /// therefore it is the last on the list. CPUs that support AVX512 will most certainly also support FMA and AVX,
  /// AVX512 takes precedence. This list should contain one entry for each of the supported CPU extensions by the
  /// FIRESTARTER payloads.
  std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>> FallbackPlatformConfigs = {
      std::make_shared<platform::SkylakeSPConfig>(),   // AVX512
      std::make_shared<platform::BulldozerConfig>(),   // FMA4
      std::make_shared<platform::HaswellConfig>(),     // FMA
      std::make_shared<platform::SandyBridgeConfig>(), // AVX
      std::make_shared<platform::NehalemConfig>()      // SSE2
  };
};

} // namespace firestarter::x86
