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

#include "../Payload/Payload.hpp"
#include "firestarter/Environment/CPUTopology.hpp"
#include "firestarter/Environment/Payload/PayloadSettings.hpp"
#include "firestarter/Logging/Log.hpp"

namespace firestarter::environment::platform {

class PlatformConfig {
private:
  std::string Name;
  payload::PayloadSettings Settings;
  std::shared_ptr<const payload::Payload> Payload;

public:
  /// Getter for the name of the platform.
  [[nodiscard]] auto name() const -> const auto& { return Name; }
  /// Getter for the settings of the platform.
  [[nodiscard]] auto settings() const -> const auto& { return Settings; }
  /// Reference to the settings. This allows them to be overriden.
  [[nodiscard]] auto settings() -> auto& { return Settings; }
  /// Getter for the payload of the platform.
  [[nodiscard]] auto payload() const -> const auto& { return Payload; }

  [[nodiscard]] auto isAvailable(const CPUTopology& Topology) const -> bool { return isAvailable(&Topology); }

  [[nodiscard]] auto isDefault(const CPUTopology& Topology) const -> bool { return isDefault(&Topology); }

protected:
  [[nodiscard]] virtual auto isAvailable(const CPUTopology* Topology) const -> bool {
    return payload()->isAvailable(*Topology);
  }

  [[nodiscard]] virtual auto isDefault(const CPUTopology*) const -> bool = 0;

public:
  PlatformConfig() = delete;

  PlatformConfig(std::string Name, payload::PayloadSettings&& Settings,
                 std::shared_ptr<const payload::Payload>&& Payload) noexcept
      : Name(std::move(Name))
      , Settings(std::move(Settings))
      , Payload(std::move(Payload)) {}

  virtual ~PlatformConfig() = default;

  /// Clone a the platform config.
  [[nodiscard]] virtual auto clone() const -> std::unique_ptr<PlatformConfig> = 0;

  /// Clone a concreate platform config.
  /// \arg InstructionCacheSize The detected size of the instructions cache.
  /// \arg ThreadPerCore The number of threads per pysical CPU.
  [[nodiscard]] virtual auto cloneConcreate(std::optional<unsigned> InstructionCacheSize, unsigned ThreadsPerCore) const
      -> std::unique_ptr<PlatformConfig> = 0;

  /// The function name for this platform config given a specific thread per core count.
  /// \arg ThreadsPerCore The number of threads per core.
  /// \returns The name of the function (a platform name, payload name and a specific thread per core count)
  [[nodiscard]] auto functionName(unsigned ThreadsPerCore) const -> std::string {
    return "FUNC_" + Name + "_" + Payload->name() + "_" + std::to_string(ThreadsPerCore) + "T";
  };

  /// Get the concreate functions name.
  [[nodiscard]] auto functionName() const -> std::string {
    assert(Settings.isConcreate() && "Settings must be concreate for a concreate function name");
    return functionName(Settings.thread());
  };

  void printCodePathSummary() const {
    assert(Settings.isConcreate() && "Setting must be concreate to print the code path summary.");

    log::info() << "\n"
                << "  Taking " << Payload->name() << " path optimized for " << Name << " - " << Settings.thread()
                << " thread(s) per core\n"
                << "  Used buffersizes per thread:";

    if (Settings.instructionCacheSizePerThread()) {
      log::info() << "    - L1i-Cache: " << *Settings.instructionCacheSizePerThread() << " Bytes";
    }

    unsigned I = 1;
    for (auto const& Bytes : Settings.dataCacheBufferSizePerThread()) {
      log::info() << "    - L" << I << "d-Cache: " << Bytes << " Bytes";
      I++;
    }

    log::info() << "    - Memory: " << Settings.ramBufferSizePerThread() << " Bytes";
  }
};

} // namespace firestarter::environment::platform
