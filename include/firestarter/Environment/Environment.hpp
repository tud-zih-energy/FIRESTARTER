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
#include "Platform/RuntimeConfig.hpp"
#include <cassert>
#include <cstdint>
#include <vector>

namespace firestarter::environment {

class Environment {
public:
  Environment() = delete;
  explicit Environment(std::unique_ptr<CPUTopology>&& Topology)
      : Topology(std::move(Topology)) {}
  virtual ~Environment() { delete SelectedConfig; }

  auto evaluateCpuAffinity(unsigned RequestedNumThreads, std::string CpuBind) -> int;
  auto setCpuAffinity(unsigned Thread) -> int;
  void printThreadSummary();

  virtual void evaluateFunctions() = 0;
  virtual auto selectFunction(unsigned FunctionId, bool AllowUnavailablePayload) -> int = 0;
  virtual auto selectInstructionGroups(std::string Groups) -> int = 0;
  virtual void printAvailableInstructionGroups() = 0;
  virtual void setLineCount(unsigned LineCount) = 0;
  virtual void printSelectedCodePathSummary() = 0;
  virtual void printFunctionSummary() = 0;

  [[nodiscard]] auto selectedConfig() const -> platform::RuntimeConfig& {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    assert(("No RuntimeConfig selected", SelectedConfig != nullptr));
#pragma GCC diagnostic pop
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return *SelectedConfig;
  }

  [[nodiscard]] auto requestedNumThreads() const -> uint64_t { return RequestedNumThreads; }

  [[nodiscard]] auto topology() const -> CPUTopology const& {
    assert(Topology != nullptr);
    return *Topology;
  }

protected:
  platform::RuntimeConfig* SelectedConfig = nullptr;
  std::unique_ptr<CPUTopology> Topology;

private:
  uint64_t RequestedNumThreads = 0;

  // TODO: replace these functions with the builtins one from hwloc
  auto cpuAllowed(unsigned Id) -> int;
  auto cpuSet(unsigned Id) -> int;

  std::vector<unsigned> CpuBind;
};

} // namespace firestarter::environment
