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

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace firestarter::environment {

/// This class handles parsing of user input to FIRESTARTER, namely the number of threads used, the thread affinity, the
/// selection of the correct high-load function, selection of the instruction groups and number of lines. It also
/// handles printing useful information, provides interfaces to the PlatformConfig and the number of threads. It
/// facilitates setting the cpu affinity in further parts of FIRESTARTER.
class Environment {
public:
  Environment() = delete;
  explicit Environment(std::unique_ptr<CPUTopology>&& Topology)
      : Topology(std::move(Topology)) {}
  virtual ~Environment() = default;

  /// Parse the user input for the cpu affinity and the number of requested threads. If a CpuBind is provided we
  /// evaluate it and set the number of threads and their affinity accordingly. This is only supported on linux and with
  /// the FIRESTARTER_THREAD_AFFINITY build flag. This function will save the correct number of threads based on the
  /// user input in RequestedNumThreads. It must be called for FIRESTARTER to function properly.
  /// \arg RequestedNumThreads The optional number of threads that are requested by a user. If this is std::nullopt the
  /// number will be automatically determined.
  /// \arg CpuBinding The optional vector of cpus which are used for FIRESTARTER. It overrides the requested number of
  /// threads.
  void evaluateCpuAffinity(const std::optional<unsigned>& RequestedNumThreads,
                           const std::optional<std::vector<uint64_t>>& CpuBinding);

  /// The worker threads are numerated from zero to RequestedNumThreads. Set the cpuaffinity of a calling thread based
  /// on this index to the one that that should be used according to the determined CpuBind list from the call to
  /// evaluateCpuAffinity. This function will throw if it is called with an invalid index.
  /// \arg Thread The index of the worker thread.
  void setCpuAffinity(unsigned Thread) const;

  /// Print the summary of the used thread for the workers. If thread affinity is supported (linux and compiled with the
  /// FIRESTARTER_THREAD_AFFINITY flag), print which thread is pinned to which CPU.
  void printThreadSummary();

  /// Select a PlatformConfig based on its generated id. This function will throw if a payload is not available or the
  /// id is incorrect. If id is zero we automatically select a matching PlatformConfig.
  /// \arg FunctionId The id of the PlatformConfig that should be selected.
  /// \arg AllowUnavailablePayload If true we will not throw if the PlatformConfig is not available.
  virtual void selectFunction(unsigned FunctionId, bool AllowUnavailablePayload) = 0;

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

  /// Get the number of threads FIRESTARTER will run with.
  [[nodiscard]] auto requestedNumThreads() const -> uint64_t { return RequestedNumThreads; }

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

  /// Const getter for the current CPU topology.
  [[nodiscard]] virtual auto topology() const -> const CPUTopology& {
    assert(Topology && "Topology is a nullptr");
    return *Topology;
  }

protected:
  /// This function sets the config based on the
  void setConfig(std::unique_ptr<platform::PlatformConfig>&& Config) { this->Config = std::move(Config); }

private:
  /// The selected config that contains the payload, settings and the associated name.
  std::unique_ptr<platform::PlatformConfig> Config;
  /// The description of the current CPU.
  std::unique_ptr<CPUTopology> Topology;

  /// The number of threads FIRESTARTER is requested to run with. This will initially be set to zero, which will be
  /// replaced by the maximum number of threads after calling evaluateCpuAffinity.
  uint64_t RequestedNumThreads = 0;

  // TODO(Issue #74): Use hwloc for cpu thread affinity.
#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  /// Check if the Cpu is allowed to be used with the current program.
  /// \arg Id The if of the CPU which is checked.
  /// \returns true if the CPU with Id is allowed to be used by the program.
  static auto cpuAllowed(unsigned Id) -> bool;

  /// Set the cpu affinity of the current thread to a specific CPU.
  /// \arg Id The id of the CPU to which to pin the calling thread.
  /// \returns 0 on success. See the man page for. sched_setaffinity.
  static auto cpuSet(unsigned Id) -> int;

  /// Add a CPU to mask if this CPU is available on the current system or throw with an error.
  /// \arg Cpu The id of the CPU to add to the mask.
  /// \arg Mask The reference to the mask to add the cpu to.
  void addCpuSet(unsigned Cpu, cpu_set_t& Mask) const;

  /// The list of physical CPU ids that are requested to be used. The length of this list should match the number of
  /// requested threads if it is not zero.
  std::vector<unsigned> CpuBind;
#endif
};

} // namespace firestarter::environment
