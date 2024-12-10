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

#include "firestarter/Platform/PlatformConfig.hpp"
#include "firestarter/ProcessorInformation.hpp"
#include "firestarter/X86/X86ProcessorInformation.hpp"

namespace firestarter::x86::platform {

/// Models a platform config that is the default based on x86 CPU family and model ids.
class X86PlatformConfig : public firestarter::platform::PlatformConfig {
private:
  /// The famility id of the processor for which this is the default platform config.
  unsigned Family;
  /// The list of model ids in combination with the family for which this is the default platform config.
  std::list<unsigned> Models;

public:
  X86PlatformConfig(std::string Name, unsigned Family, std::list<unsigned>&& Models,
                    firestarter::payload::PayloadSettings&& Settings,
                    std::shared_ptr<const firestarter::payload::Payload>&& Payload) noexcept
      : PlatformConfig(std::move(Name), std::move(Settings), std::move(Payload))
      , Family(Family)
      , Models(std::move(Models)) {}

  /// Check if this platform is available on the current system. This transloate to if the cpu extensions are
  /// available for the payload that is used.
  /// \arg Topology The reference to the X86CPUTopology that is used to check agains if this platform is supported.
  /// \returns true if the platform is supported on the given X86CPUTopology.
  [[nodiscard]] auto isAvailable(const X86ProcessorInformation& Topology) const -> bool {
    return isAvailable(&Topology);
  }

  /// Check if this platform is available and the default on the current system.
  /// \arg Topology The reference to the X86CPUTopology that is used to check agains if this payload is supported.
  /// \returns true if the platform is the default one for a given X86CPUTopology.
  [[nodiscard]] auto isDefault(const X86ProcessorInformation& Topology) const -> bool { return isDefault(&Topology); }

  /// Clone a the platform config.
  [[nodiscard]] auto clone() const -> std::unique_ptr<PlatformConfig> final {
    auto Ptr = std::make_unique<X86PlatformConfig>(name(), Family, std::list<unsigned>(Models),
                                                   firestarter::payload::PayloadSettings(settings()),
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
  /// Check if this platform is available on the current system. This tranlates to if the cpu extensions are
  /// available for the payload that is used.
  /// \arg Topology The pointer to the CPUTopology that is used to check agains if this platform is supported.
  /// \returns true if the platform is supported on the given CPUTopology.
  [[nodiscard]] auto isAvailable(const ProcessorInformation* Topology) const -> bool final {
    return firestarter::platform::PlatformConfig::isAvailable(Topology);
  }

  /// Check if this platform is available and the default on the current system. This is done by checking if the family
  /// id in the CPUTopology matches the one saved in Family and if the model id in the CPUTopology is contained in
  /// Models.
  /// \arg Topology The pointer to the CPUTopology that is used to check agains if this payload is supported.
  /// \returns true if the platform is the default one for a given CPUTopology.
  [[nodiscard]] auto isDefault(const ProcessorInformation* Topology) const -> bool final {
    const auto* FinalTopology = dynamic_cast<const X86ProcessorInformation*>(Topology);
    assert(FinalTopology && "isDefault not called with const X86CPUTopology*");

    // Check if the family of the topology matches the family of the config, if the model of the topology is contained
    // in the models list of the config and if the config is available on the current platform.
    return Family == FinalTopology->familyId() &&
           (std::find(Models.begin(), Models.end(), FinalTopology->modelId()) != Models.end()) && isAvailable(Topology);
  }
};

} // namespace firestarter::x86::platform
