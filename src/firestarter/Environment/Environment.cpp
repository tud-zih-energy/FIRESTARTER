/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
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

#include "firestarter/Environment/Environment.hpp"
#include "firestarter/Logging/Log.hpp"

#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>

namespace firestarter::environment {

#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)

extern "C" {
#include <sched.h>
}

auto Environment::cpuSet(unsigned Id) -> int {
  cpu_set_t Mask;

  CPU_ZERO(&Mask);
  CPU_SET(Id, &Mask);

  return sched_setaffinity(0, sizeof(cpu_set_t), &Mask);
}

auto Environment::cpuAllowed(unsigned Id) -> bool {
  cpu_set_t Mask;

  CPU_ZERO(&Mask);

  if (!sched_getaffinity(0, sizeof(cpu_set_t), &Mask)) {
    return CPU_ISSET(Id, &Mask);
  }

  return false;
}

void Environment::addCpuSet(unsigned Cpu, cpu_set_t& Mask) const {
  if (cpuAllowed(Cpu)) {
    CPU_SET(Cpu, &Mask);
  } else {
    if (Cpu >= topology().numThreads()) {
      throw std::invalid_argument("The given bind argument (-b/--bind) includes CPU " + std::to_string(Cpu) +
                                  " that is not available on this system.");
    }
    throw std::invalid_argument("The given bind argument (-b/--bind) cannot "
                                "be implemented with the cpuset given from the OS\n"
                                "This can be caused by the taskset tool, cgroups, "
                                "the batch system, or similar mechanisms.\n"
                                "Please fix the argument to match the restrictions.");
  }
}
#endif

void Environment::evaluateCpuAffinity(const std::optional<unsigned>& RequestedNumThreads,
                                      const std::optional<std::vector<uint64_t>>& CpuBinding) {
  if (RequestedNumThreads && RequestedNumThreads > topology().numThreads()) {
    log::warn() << "Not enough CPUs for requested number of threads";
  }

#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  cpu_set_t Cpuset;

  CPU_ZERO(&Cpuset);

  unsigned NumThreads = 0;

  if (RequestedNumThreads) {
    // RequestedNumThreads provided, pin to the first *RequestedNumThreads CPUs.
    for (unsigned I = 0; I < topology().maxNumThreads() && NumThreads < *RequestedNumThreads; I++) {
      // skip if cpu is not available
      if (!cpuAllowed(I)) {
        continue;
      }
      addCpuSet(I, Cpuset);
      NumThreads++;
    }
    // requested to many threads
    if (NumThreads < *RequestedNumThreads) {
      throw std::invalid_argument("You are requesting more threads than "
                                  "there are CPUs available in the given cpuset.\n"
                                  "This can be caused by the taskset tool, cgrous, "
                                  "the batch system, or similar mechanisms.\n"
                                  "Please fix the -n/--threads argument to match the "
                                  "restrictions.");
    }
  } else if (CpuBinding) {
    // CpuBinding provided, pin to the specified cpus
    for (const auto& CpuId : *CpuBinding) {
      addCpuSet(CpuId, Cpuset);
    }
    NumThreads = CpuBinding->size();
  } else {
    // Neither RequestedNumThreads nor CpuBinding provided, pin to all available CPUs.
    for (unsigned I = 0; I < topology().maxNumThreads(); I++) {
      if (cpuAllowed(I)) {
        addCpuSet(I, Cpuset);
        NumThreads++;
      }
    }
  }

  // Save the ids of the threads.
  for (unsigned I = 0; I < topology().maxNumThreads(); I++) {
    if (CPU_ISSET(I, &Cpuset)) {
      this->CpuBind.push_back(I);
    }
  }

  this->RequestedNumThreads = NumThreads;
#else
  (void)CpuBinding;

  if (!RequestedNumThreads) {
    this->RequestedNumThreads = topology().maxNumThreads();
  } else {
    // Limit the number of thread to the maximum on the CPU.
    this->RequestedNumThreads = (std::min)(*RequestedNumThreads, topology().maxNumThreads());
  }
#endif
}

void Environment::printThreadSummary() {
  log::info() << "\n  using " << requestedNumThreads() << " threads";

#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  bool PrintCoreIdInfo = false;
  size_t I = 0;

  std::vector<unsigned> CpuBind(this->CpuBind);
  CpuBind.resize(requestedNumThreads());
  for (auto const& Bind : CpuBind) {
    const auto CoreId = topology().getCoreIdFromPU(Bind);
    const auto PkgId = topology().getPkgIdFromPU(Bind);

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
#endif
}

void Environment::setCpuAffinity(unsigned Thread) const {
  if (Thread >= requestedNumThreads()) {
    throw std::invalid_argument("Trying to set more CPUs than available.");
  }

#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  cpuSet(CpuBind.at(Thread));
#endif
}
}; // namespace firestarter::environment