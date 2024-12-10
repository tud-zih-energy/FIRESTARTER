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

#include "firestarter/Logging/Log.hpp"
#include "firestarter/Payload/Payload.hpp"
#include "firestarter/ProcessorInformation.hpp"

#include <algorithm>
#include <cstdio>
#include <regex>

namespace firestarter::platform {

/// The payload in combination with settings and a short hand name for the specific microarchitecture this payload is
/// designed for.
class PlatformConfig {
private:
  /// The name of this platform. This is usually a short hand for the CPU microarchitecture e.g., HSW_COREI or
  /// HSW_XEONEP.
  std::string Name;

  /// The settings for the associated payload.
  payload::PayloadSettings Settings;

  /// The payload this platfrom should execute.
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

  /// Check if this platform is available on the current system. This transloate to if the cpu extensions are
  /// available for the payload that is used.
  /// \arg Topology The reference to the CPUTopology that is used to check agains if this platform is supported.
  /// \returns true if the platform is supported on the given CPUTopology.
  [[nodiscard]] auto isAvailable(const ProcessorInformation& Topology) const -> bool { return isAvailable(&Topology); }

  /// Check if this platform is available and the default on the current system.
  /// \arg Topology The reference to the CPUTopology that is used to check agains if this payload is supported.
  /// \returns true if the platform is the default one for a given CPUTopology.
  [[nodiscard]] auto isDefault(const ProcessorInformation& Topology) const -> bool { return isDefault(&Topology); }

protected:
  /// Check if this platform is available on the current system. This transloate to if the cpu extensions are
  /// available for the payload that is used.
  /// \arg Topology The pointer to the CPUTopology that is used to check agains if this platform is supported.
  /// \returns true if the platform is supported on the given CPUTopology.
  [[nodiscard]] virtual auto isAvailable(const ProcessorInformation* Topology) const -> bool {
    return payload()->isAvailable(*Topology);
  }

  /// Check if this platform is available and the default on the current system.
  /// \arg Topology The pointer to the CPUTopology that is used to check agains if this payload is supported.
  /// \returns true if the platform is the default one for a given CPUTopology.
  [[nodiscard]] virtual auto isDefault(const ProcessorInformation*) const -> bool = 0;

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

  /// Parse the selected payload instruction groups and save the in the selected function. Throws if the input is
  /// invalid.
  /// \arg Groups The list of instruction groups that is in the format: multiple INSTRUCTION:VALUE pairs
  /// comma-seperated.
  void selectInstructionGroups(const std::string& Groups) {
    const auto Delimiter = ',';
    const std::regex Re("^(\\w+):(\\d+)$");
    const auto AvailableInstructionGroups = payload()->getAvailableInstructions();

    std::stringstream Ss(Groups);
    std::vector<std::pair<std::string, unsigned>> PayloadSettings = {};

    while (Ss.good()) {
      std::string Token;
      std::smatch M;
      std::getline(Ss, Token, Delimiter);

      if (std::regex_match(Token, M, Re)) {
        if (std::find(AvailableInstructionGroups.begin(), AvailableInstructionGroups.end(), M[1].str()) ==
            AvailableInstructionGroups.end()) {
          throw std::invalid_argument("Invalid instruction-group: " + M[1].str() +
                                      "\n       --run-instruction-groups format: multiple INST:VAL "
                                      "pairs comma-seperated");
        }
        auto Num = std::stoul(M[2].str());
        if (Num == 0) {
          throw std::invalid_argument("instruction-group VAL may not contain number 0"
                                      "\n       --run-instruction-groups format: multiple INST:VAL "
                                      "pairs comma-seperated");
        }
        PayloadSettings.emplace_back(M[1].str(), Num);
      } else {
        throw std::invalid_argument("Invalid symbols in instruction-group: " + Token +
                                    "\n       --run-instruction-groups format: multiple INST:VAL "
                                    "pairs comma-seperated");
      }
    }

    settings().selectInstructionGroups(PayloadSettings);

    log::info() << "  Running custom instruction group: " << Groups;
  }

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

  /// Print a summary for the selected platform/payload with given settings.
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

} // namespace firestarter::platform
