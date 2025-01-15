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

#include "firestarter/CPUTopology.hpp"
#include "firestarter/Platform/PlatformConfig.hpp"
#include "firestarter/ProcessorInformation.hpp"

#include <cassert>
#include <memory>

namespace firestarter {

class FunctionSelection {
public:
  FunctionSelection() = default;
  virtual ~FunctionSelection() = default;

  /// Select a PlatformConfig based on its generated id. This function will throw if a payload is not available or the
  /// id is incorrect.
  /// \arg FunctionId The id of the PlatformConfig that should be selected or automatically select a matching
  /// PlatformConfig.
  /// \arg ProcessorInfos Information about the processor which is specific to the current ISA.
  /// \arg Topology The topology which contains information about the cpu requied to select the correct function.
  /// \arg AllowUnavailablePayload If true we will not throw if the PlatformConfig is not available.
  [[nodiscard]] auto selectFunction(std::optional<unsigned> FunctionId, const ProcessorInformation& ProcessorInfos,
                                    const CPUTopology& Topology,
                                    bool AllowUnavailablePayload) const -> std::unique_ptr<platform::PlatformConfig>;

  /// Select a PlatformConfig based on its generated id. This function will throw if a payload is not available or the
  /// id is incorrect.
  /// \arg FunctionId The id of the PlatformConfig that should be selected.
  /// \arg Features The CPU features of the current processor.
  /// \arg InstructionCacheSize The optional size of the instruction cache.
  /// \arg AllowUnavailablePayload If true we will not throw if the PlatformConfig is not available.
  [[nodiscard]] auto
  selectAvailableFunction(unsigned FunctionId, const CpuFeatures& Features,
                          std::optional<unsigned> InstructionCacheSize,
                          bool AllowUnavailablePayload) const -> std::unique_ptr<platform::PlatformConfig>;

  /// Select the fallback PlatformConfig if no id is given.
  /// \arg Model The class that identifies the cpu model.
  /// \arg Features The CPU features of the current processor.
  /// \arg VendorString The string of the cpu vendor.
  /// \arg ModelString The string of the cpu model.
  /// \arg InstructionCacheSize The optional size of the instruction cache.
  /// \arg NumThreadsPerCore The number of threads per core.
  [[nodiscard]] auto
  selectDefaultOrFallbackFunction(const CpuModel& Model, const CpuFeatures& Features, const std::string& VendorString,
                                  const std::string& ModelString, std::optional<unsigned> InstructionCacheSize,
                                  unsigned NumThreadsPerCore) const -> std::unique_ptr<platform::PlatformConfig>;

  /// Print a list of available high-load function and if they are available on the current system.
  /// \arg ProcessorInfos Information about the processor which is specific to the current ISA.
  /// \arg ForceYes Force all functions to be shown as avaialable
  void printFunctionSummary(const ProcessorInformation& ProcessorInfos, bool ForceYes) const;

  [[nodiscard]] virtual auto
  platformConfigs() const -> const std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>>& = 0;

  [[nodiscard]] virtual auto
  fallbackPlatformConfigs() const -> const std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>>& = 0;
};

} // namespace firestarter
