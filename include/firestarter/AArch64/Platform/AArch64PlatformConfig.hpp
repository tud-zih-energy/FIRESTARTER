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

#include "firestarter/AArch64/AArch64CpuModel.hpp"
#include "firestarter/Platform/PlatformConfig.hpp"

#include <set>

namespace firestarter::aarch64::platform {

/// Models a platform config that is the default based on AArch64 CPU model.
class AArch64PlatformConfig : public firestarter::platform::PlatformConfig {
private:
  /// The set of requested cpu models
  std::set<AArch64CpuModel> RequestedModels;

public:
  AArch64PlatformConfig(std::string Name, std::set<AArch64CpuModel>&& RequestedModels,
                        firestarter::payload::PayloadSettings&& Settings,
                        std::shared_ptr<const firestarter::payload::Payload>&& Payload) noexcept
      : PlatformConfig(std::move(Name), std::move(Settings), std::move(Payload))
      , RequestedModels(std::move(RequestedModels)) {}

  /// Clone a the platform config.
  [[nodiscard]] auto clone() const -> std::unique_ptr<PlatformConfig> final {
    auto Ptr = std::make_unique<AArch64PlatformConfig>(name(), std::set<AArch64CpuModel>(RequestedModels),
                                                       firestarter::payload::PayloadSettings(settings()),
                                                       std::shared_ptr(payload()));
    return Ptr;
  }

  /// Clone a concrete platform config.
  [[nodiscard]] auto cloneConcreate(std::optional<unsigned> InstructionCacheSize, unsigned ThreadsPerCore) const
      -> std::unique_ptr<PlatformConfig> final {
    auto Ptr = clone();
    auto* DerivedPtr = dynamic_cast<AArch64PlatformConfig*>(Ptr.get());
    DerivedPtr->settings().concretize(InstructionCacheSize, ThreadsPerCore);
    return Ptr;
  }

  /// Check if this platform is available and the default on the current system.
  [[nodiscard]] auto isDefault(const CpuModel& Model, const CpuFeatures& Features) const -> bool override {
    const auto ModelIt = std::find(RequestedModels.cbegin(), RequestedModels.cend(), Model);
    return ModelIt != RequestedModels.cend() && payload()->isAvailable(Features);
  }
};

} // namespace firestarter::aarch64::platform
