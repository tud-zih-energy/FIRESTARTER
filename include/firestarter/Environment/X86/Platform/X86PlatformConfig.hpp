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

#include "../../Platform/PlatformConfig.hpp"
#include "firestarter/Environment/CPUTopology.hpp"
#include "firestarter/Environment/X86/X86CPUTopology.hpp"

namespace firestarter::environment::x86::platform {

class X86PlatformConfig : public environment::platform::PlatformConfig {
private:
  unsigned Family;
  std::list<unsigned> Models;

public:
  X86PlatformConfig(std::string Name, unsigned Family, std::list<unsigned>&& Models,
                    environment::payload::PayloadSettings&& Settings,
                    std::shared_ptr<const environment::payload::Payload>&& Payload) noexcept
      : PlatformConfig(std::move(Name), std::move(Settings), std::move(Payload))
      , Family(Family)
      , Models(std::move(Models)) {}

  [[nodiscard]] auto isAvailable(const X86CPUTopology& Topology) const -> bool { return isAvailable(&Topology); }

  [[nodiscard]] auto isDefault(const X86CPUTopology& Topology) const -> bool { return isDefault(&Topology); }

  /// Clone a the platform config.
  [[nodiscard]] auto clone() const -> std::unique_ptr<PlatformConfig> final {
    auto Ptr = std::make_unique<X86PlatformConfig>(name(), Family, std::list<unsigned>(Models),
                                                   environment::payload::PayloadSettings(settings()),
                                                   std::shared_ptr(payload()));
    return Ptr;
  }

  /// Clone a concreate platform config.
  /// \arg InstructionCacheSize The detected size of the instructions cache.
  /// \arg ThreadPerCore The number of threads per pysical CPU.
  [[nodiscard]] auto cloneConcreate(std::optional<unsigned> InstructionCacheSize, unsigned ThreadsPerCore) const
      -> std::unique_ptr<PlatformConfig> final {
    auto Ptr = clone();
    Ptr->settings().concretize(InstructionCacheSize, ThreadsPerCore);
    return Ptr;
  }

private:
  [[nodiscard]] auto isAvailable(const CPUTopology* Topology) const -> bool final {
    return environment::platform::PlatformConfig::isAvailable(Topology);
  }

  [[nodiscard]] auto isDefault(const CPUTopology* Topology) const -> bool final {
    const auto* FinalTopology = dynamic_cast<const X86CPUTopology*>(Topology);
    assert(FinalTopology && "isDefault not called with const X86CPUTopology*");

    // Check if the family of the topology matches the family of the config, if the model of the topology is contained
    // in the models list of the config and if the config is available on the current platform.
    return Family == FinalTopology->familyId() &&
           (std::find(Models.begin(), Models.end(), FinalTopology->modelId()) != Models.end()) && isAvailable(Topology);
  }
};

} // namespace firestarter::environment::x86::platform
