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

#include "firestarter/Environment/Environment.hpp"
#include "firestarter/Environment/X86/Platform/BulldozerConfig.hpp"
#include "firestarter/Environment/X86/Platform/HaswellConfig.hpp"
#include "firestarter/Environment/X86/Platform/HaswellEPConfig.hpp"
#include "firestarter/Environment/X86/Platform/KnightsLandingConfig.hpp"
#include "firestarter/Environment/X86/Platform/NaplesConfig.hpp"
#include "firestarter/Environment/X86/Platform/NehalemConfig.hpp"
#include "firestarter/Environment/X86/Platform/NehalemEPConfig.hpp"
#include "firestarter/Environment/X86/Platform/RomeConfig.hpp"
#include "firestarter/Environment/X86/Platform/SandyBridgeConfig.hpp"
#include "firestarter/Environment/X86/Platform/SandyBridgeEPConfig.hpp"
#include "firestarter/Environment/X86/Platform/SkylakeConfig.hpp"
#include "firestarter/Environment/X86/Platform/SkylakeSPConfig.hpp"
#include "firestarter/Environment/X86/Platform/X86PlatformConfig.hpp"

namespace firestarter::environment::x86 {

class X86Environment final : public Environment {
public:
  X86Environment()
      : Environment(std::make_unique<X86ProcessorInformation>()) {}

  /// Getter (which allows modifying) for the current platform config containing the payload, settings, the
  /// associated name and the default X86 family and models.
  [[nodiscard]] auto config() -> platform::X86PlatformConfig& final {
    auto* X86PlatformConfig = dynamic_cast<platform::X86PlatformConfig*>(&Environment::config());
    assert(X86PlatformConfig && "X86PlatformConfig is a nullptr");
    return *X86PlatformConfig;
  }

  /// Const getter for the current platform config containing the payload, settings, the associated name and the default
  /// X86 family and models.
  [[nodiscard]] auto config() const -> const platform::X86PlatformConfig& final {
    const auto* X86PlatformConfig = dynamic_cast<const platform::X86PlatformConfig*>(&Environment::config());
    assert(X86PlatformConfig && "X86PlatformConfig is a nullptr");
    return *X86PlatformConfig;
  }

  /// Const getter for the current CPU topology with X86 specific modifications.
  [[nodiscard]] auto topology() const -> const X86ProcessorInformation& final {
    const auto* X86Topology = dynamic_cast<const X86ProcessorInformation*>(&Environment::topology());
    assert(X86Topology && "X86Topology is a nullptr");
    return *X86Topology;
  }

  /// Select a PlatformConfig based on its generated id. This function will throw if a payload is not available or the
  /// id is incorrect. If id is zero we automatically select a matching PlatformConfig.
  /// \arg FunctionId The id of the PlatformConfig that should be selected.
  /// \arg AllowUnavailablePayload If true we will not throw if the PlatformConfig is not available.
  void selectFunction(unsigned FunctionId, bool AllowUnavailablePayload) override;

  /// Parse the selected payload instruction groups and save the in the selected function. Throws if the input is
  /// invalid.
  /// \arg Groups The list of instruction groups that is in the format: multiple INSTRUCTION:VALUE pairs
  /// comma-seperated.
  void selectInstructionGroups(std::string Groups) override;

  /// Print the available instruction groups of the selected function.
  void printAvailableInstructionGroups() override;

  /// Set the line count in the selected function.
  /// \arg LineCount The maximum number of instruction that should be in the high-load loop.
  void setLineCount(unsigned LineCount) override;

  /// Print a summary of the settings of the selected config.
  void printSelectedCodePathSummary() override;

  /// Print a list of available high-load function and if they are available on the current system. This includes all
  /// PlatformConfigs in combination with all thread per core counts.
  /// \arg ForceYes Force all functions to be shown as avaialable
  void printFunctionSummary(bool ForceYes) override;

private:
  /// The list of availabe platform configs that is printed when supplying the --avail command line argument. The IDs
  /// for these configs are generated by iterating through this list starting with 1. To maintain stable IDs in
  /// FIRESTARTER new configs should be added to the bottom of the list.
  const std::list<std::shared_ptr<platform::X86PlatformConfig>> PlatformConfigs = {
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
  const std::list<std::shared_ptr<platform::X86PlatformConfig>> FallbackPlatformConfigs = {
      std::make_shared<platform::SkylakeSPConfig>(),   // AVX512
      std::make_shared<platform::BulldozerConfig>(),   // FMA4
      std::make_shared<platform::HaswellConfig>(),     // FMA
      std::make_shared<platform::SandyBridgeConfig>(), // AVX
      std::make_shared<platform::NehalemConfig>()      // SSE2
  };
};

} // namespace firestarter::environment::x86
