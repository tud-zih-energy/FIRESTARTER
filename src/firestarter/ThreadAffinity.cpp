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

#include "firestarter/ThreadAffinity.hpp"
#include "firestarter/CPUTopology.hpp"
#include "firestarter/Logging/Log.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>

namespace firestarter {

auto ThreadAffinity::fromCommandLine(const HardwareThreadsInfo& ThreadsInfo,
                                     const std::optional<unsigned>& RequestedNumThreads,
                                     const std::optional<std::set<uint64_t>>& CpuBinding) -> ThreadAffinity {
  ThreadAffinity Affinity{};

  if (RequestedNumThreads && (*RequestedNumThreads > ThreadsInfo.MaxNumThreads)) {
    log::warn() << "Not enough CPUs for requested number of threads";
  }

  if (RequestedNumThreads) {
    // RequestedNumThreads provided, pin to the first *RequestedNumThreads CPUs.
    for (const auto& OsIndex : ThreadsInfo.OsIndices) {
      if (Affinity.RequestedNumThreads == *RequestedNumThreads) {
        break;
      }
      Affinity.CpuBind.emplace(OsIndex);
      Affinity.RequestedNumThreads++;
    }
    // requested too many threads
    if (Affinity.RequestedNumThreads < *RequestedNumThreads) {
      throw std::invalid_argument("You are requesting more threads than "
                                  "there are CPUs available in the given cpuset.\n"
                                  "This can be caused by the taskset tool, cgroups, "
                                  "the batch system, or similar mechanisms.\n"
                                  "Please fix the -n/--threads argument to match the "
                                  "restrictions.");
    }
  } else if (CpuBinding) {
    // CpuBinding provided, pin to the specified cpus
    const auto& AllowedOsIndices = ThreadsInfo.OsIndices;

    for (const auto& OsIndex : *CpuBinding) {
      if (AllowedOsIndices.find(OsIndex) == AllowedOsIndices.cend()) {
        // Id is not allowed
        throw std::invalid_argument("The given bind argument (-b/--bind) cannot "
                                    "be implemented with the cpuset given from the OS\n"
                                    "This can be caused by the taskset tool, cgroups, "
                                    "the batch system, or similar mechanisms.\n"
                                    "Please fix the argument to match the restrictions.");
      }
      Affinity.CpuBind.emplace(OsIndex);
    }
    Affinity.RequestedNumThreads = CpuBinding->size();
  } else {
    // Neither RequestedNumThreads nor CpuBinding provided, pin to all available CPUs.
    for (const auto& OsIndex : ThreadsInfo.OsIndices) {
      Affinity.CpuBind.emplace(OsIndex);
      Affinity.RequestedNumThreads++;
    }
  }

  return Affinity;
}

void ThreadAffinity::printThreadSummary(const CPUTopology& Topology) const {
  log::info() << "\n  using " << RequestedNumThreads << " threads";

  bool PrintCoreIdInfo = false;
  size_t I = 0;

  for (auto const& Bind : CpuBind) {
    const auto CoreId = Topology.getCoreIdFromPU(Bind);
    const auto PkgId = Topology.getPkgIdFromPU(Bind);

    if (CoreId && PkgId) {
      log::info() << "    - Thread " << I << " run on CPU " << Bind << ", core " << *CoreId
                  << " in package: " << *PkgId;
      PrintCoreIdInfo = true;
    }

    I++;
  }

  if (PrintCoreIdInfo) {
    log::info() << "  The cores are numbered using the logical_index from hwloc.";
  }
}

}; // namespace firestarter