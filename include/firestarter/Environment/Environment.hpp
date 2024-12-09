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

#include "firestarter/Environment/CPUTopology.hpp"
#include "firestarter/Environment/Platform/PlatformConfig.hpp"
#include "firestarter/Environment/ProcessorInformation.hpp"

#include <cassert>
#include <memory>

namespace firestarter::environment {

/// This class handles parsing of user input to FIRESTARTER, namely the number of threads used, the thread affinity, the
/// selection of the correct high-load function, selection of the instruction groups and number of lines. It also
/// handles printing useful information, provides interfaces to the PlatformConfig and the number of threads. It
/// facilitates setting the cpu affinity in further parts of FIRESTARTER.
class Environment {
public:
  Environment() = delete;
  explicit Environment(std::unique_ptr<ProcessorInformation>&& Topology)
      : ProcessorInfos(std::move(Topology)) {}
  virtual ~Environment() = default;

  /// Select a PlatformConfig based on its generated id. This function will throw if a payload is not available or the
  /// id is incorrect. If id is zero we automatically select a matching PlatformConfig.
  /// \arg FunctionId The id of the PlatformConfig that should be selected.
  /// \arg Topology The topology which contains information about the cpu requied to select the correct function.
  /// \arg AllowUnavailablePayload If true we will not throw if the PlatformConfig is not available.
  virtual void selectFunction(unsigned FunctionId, const CPUTopology& Topology, bool AllowUnavailablePayload) = 0;

  /// Parse the selected payload instruction groups and save the in the selected function. Throws if the input is
  /// invalid.
  /// \arg Groups The list of instruction groups that is in the format: multiple INSTRUCTION:VALUE pairs
  /// comma-seperated.
  virtual void selectInstructionGroups(std::string Groups) = 0;

  /// Print the available instruction groups of the selected function.
  virtual void printAvailableInstructionGroups() = 0;

  /// Set the line count in the selected function.
  /// \arg LineCount The maximum number of instruction that should be in the high-load loop.
  virtual void setLineCount(unsigned LineCount) = 0;

  /// Print a summary of the settings of the selected config.
  virtual void printSelectedCodePathSummary() = 0;

  /// Print a list of available high-load function and if they are available on the current system.
  /// \arg ForceYes Force all functions to be shown as avaialable
  virtual void printFunctionSummary(bool ForceYes) = 0;

  /// Getter (which allows modifying) for the current platform config containing the payload, settings and the
  /// associated name.
  [[nodiscard]] virtual auto config() -> platform::PlatformConfig& {
    assert(Config && "No PlatformConfig selected");
    return *Config;
  }

  /// Const getter for the current platform config containing the payload, settings and the associated name.
  [[nodiscard]] virtual auto config() const -> const platform::PlatformConfig& {
    assert(Config && "No PlatformConfig selected");
    return *Config;
  }

  /// Const getter for the current processor information.
  [[nodiscard]] virtual auto processorInfos() const -> const ProcessorInformation& {
    assert(ProcessorInfos && "ProcessorInfos is a nullptr");
    return *ProcessorInfos;
  }

protected:
  /// This function sets the config based on the
  void setConfig(std::unique_ptr<platform::PlatformConfig>&& Config) { this->Config = std::move(Config); }

private:
  /// The selected config that contains the payload, settings and the associated name.
  std::unique_ptr<platform::PlatformConfig> Config;
  /// The description of the current CPU.
  std::unique_ptr<ProcessorInformation> ProcessorInfos;
};

} // namespace firestarter::environment
