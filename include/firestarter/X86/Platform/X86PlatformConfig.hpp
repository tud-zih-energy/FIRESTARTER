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
#include "firestarter/X86/X86CpuModel.hpp"
#include <set>

namespace firestarter::x86::platform {

/// Models a platform config that is the default based on x86 CPU family and model ids.
class X86PlatformConfig : public firestarter::platform::PlatformConfig {
private:
  /// The set of requested cpu models
  std::set<X86CpuModel> RequestedModels;

public:
  X86PlatformConfig(std::string Name, std::set<X86CpuModel>&& RequestedModels,
                    firestarter::payload::PayloadSettings&& Settings,
                    std::shared_ptr<const firestarter::payload::Payload>&& Payload) noexcept
      : PlatformConfig(std::move(Name), std::move(Settings), std::move(Payload))
      , RequestedModels(std::move(RequestedModels)) {}

  /// Clone a the platform config.
  [[nodiscard]] auto clone() const -> std::unique_ptr<PlatformConfig> final {
    auto Ptr = std::make_unique<X86PlatformConfig>(name(), std::set<X86CpuModel>(RequestedModels),
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
    auto* DerivedPtr = dynamic_cast<X86PlatformConfig*>(Ptr.get());
    DerivedPtr->settings().concretize(InstructionCacheSize, ThreadsPerCore);
    return Ptr;
  }

  /// Check if this platform is available and the default on the current system. This is done by checking if the cpu
  /// model matches one of the requested ones and that the payload is available with the supplied cpu features.
  /// \arg Model The reference to the cpu model that is used to check if this config is the default.
  /// \arg CpuFeatures Features that this payload requires to check agains if this payload is supported.
  /// \returns true if the platform is the default one.
  [[nodiscard]] auto isDefault(const CpuModel& Model, const CpuFeatures& Features) const -> bool override {
    const auto ModelIt = std::find(RequestedModels.cbegin(), RequestedModels.cend(), Model);
    return ModelIt != RequestedModels.cend() && payload()->isAvailable(Features);
  }
};

} // namespace firestarter::x86::platform
