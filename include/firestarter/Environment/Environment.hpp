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

#include <cassert>
#include <firestarter/Environment/CPUTopology.hpp>
#include <firestarter/Environment/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/Platform/RuntimeConfig.hpp>
#include <vector>

namespace firestarter::environment {

class Environment {
public:
  Environment(CPUTopology* topology)
      : _topology(topology) {}
  ~Environment() {
    delete this->_topology;
    if (_selectedConfig != nullptr) {
      delete _selectedConfig;
    }
  }

  int evaluateCpuAffinity(unsigned requestedNumThreads, std::string cpuBind);
  int setCpuAffinity(unsigned thread);
  void printThreadSummary();

  virtual void evaluateFunctions() = 0;
  virtual int selectFunction(unsigned functionId, bool allowUnavailablePayload) = 0;
  virtual int selectInstructionGroups(std::string groups) = 0;
  virtual void printAvailableInstructionGroups() = 0;
  virtual void setLineCount(unsigned lineCount) = 0;
  virtual void printSelectedCodePathSummary() = 0;
  virtual void printFunctionSummary() = 0;

  platform::RuntimeConfig& selectedConfig() const {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    assert(("No RuntimeConfig selected", _selectedConfig != nullptr));
#pragma GCC diagnostic pop
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return *_selectedConfig;
  }

  unsigned long long requestedNumThreads() const { return _requestedNumThreads; }

  CPUTopology const& topology() const {
    assert(_topology != nullptr);
    return *_topology;
  }

protected:
  platform::RuntimeConfig* _selectedConfig = nullptr;
  CPUTopology* _topology = nullptr;

private:
  unsigned long long _requestedNumThreads;

  // TODO: replace these functions with the builtins one from hwloc
  int cpuAllowed(unsigned id);
  int cpuSet(unsigned id);

  std::vector<unsigned> cpuBind;
};

} // namespace firestarter::environment
