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

#include "CPUTopology.hpp"
#include "firestarter/Environment/Platform/PlatformConfig.hpp"
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace firestarter::environment {

class Environment {
public:
  Environment() = delete;
  explicit Environment(std::unique_ptr<CPUTopology>&& Topology)
      : Topology(std::move(Topology)) {}
  virtual ~Environment() = default;

  void evaluateCpuAffinity(unsigned RequestedNumThreads, const std::string& CpuBind);
  void setCpuAffinity(unsigned Thread) const;
  void printThreadSummary();

  virtual void selectFunction(unsigned FunctionId, bool AllowUnavailablePayload) = 0;
  virtual void selectInstructionGroups(std::string Groups) = 0;
  virtual void printAvailableInstructionGroups() = 0;
  virtual void setLineCount(unsigned LineCount) = 0;
  virtual void printSelectedCodePathSummary() = 0;
  virtual void printFunctionSummary() = 0;

  [[nodiscard]] auto config() const -> platform::PlatformConfig& {
    assert(Config && "No PlatformConfig selected");
    return *Config;
  }

  [[nodiscard]] auto requestedNumThreads() const -> uint64_t { return RequestedNumThreads; }

  [[nodiscard]] auto topology() const -> CPUTopology const& {
    assert(Topology != nullptr && "Topology is a nullptr");
    return *Topology;
  }

protected:
  std::unique_ptr<platform::PlatformConfig> Config;
  std::unique_ptr<CPUTopology> Topology;

private:
  uint64_t RequestedNumThreads = 0;

  // TODO(Issue #74): Use hwloc for cpu thread affinity.
#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  static auto cpuAllowed(unsigned Id) -> int;
  static auto cpuSet(unsigned Id) -> int;
  void addCpuSet(unsigned Cpu, cpu_set_t& Mask) const;
#endif

  std::vector<unsigned> CpuBind;
};

} // namespace firestarter::environment
