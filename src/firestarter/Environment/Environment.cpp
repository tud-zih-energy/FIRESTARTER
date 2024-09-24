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

#include <algorithm>
#include <firestarter/Environment/Environment.hpp>
#include <firestarter/Logging/Log.hpp>

#include <regex>
#include <string>

namespace firestarter::environment {

#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)

extern "C" {
#include <sched.h>
}

// this code is from the C version of FIRESTARTER
// TODO: replace this with cpu affinity of hwloc
#define ADD_CPU_SET(cpu, cpuset)                                                                                       \
  do {                                                                                                                 \
    if (this->cpuAllowed(cpu)) {                                                                                       \
      CPU_SET(cpu, &cpuset);                                                                                           \
    } else {                                                                                                           \
      if (cpu >= this->topology().numThreads()) {                                                                      \
        log::error() << "The given bind argument (-b/--bind) includes CPU " << cpu                                     \
                     << " that is not available on this system.";                                                      \
      } else {                                                                                                         \
        log::error() << "The given bind argument (-b/--bind) cannot "                                                  \
                        "be implemented with the cpuset given from the OS\n"                                           \
                     << "This can be caused by the taskset tool, cgroups, "                                            \
                        "the batch system, or similar mechanisms.\n"                                                   \
                     << "Please fix the argument to match the restrictions.";                                          \
      }                                                                                                                \
      return EACCES;                                                                                                   \
    }                                                                                                                  \
  } while (0)

auto Environment::cpuSet(unsigned Id) -> int {
  cpu_set_t Mask;

  CPU_ZERO(&Mask);
  CPU_SET(Id, &Mask);

  return sched_setaffinity(0, sizeof(cpu_set_t), &Mask);
}

auto Environment::cpuAllowed(unsigned Id) -> int {
  cpu_set_t Mask;

  CPU_ZERO(&Mask);

  if (!sched_getaffinity(0, sizeof(cpu_set_t), &Mask)) {
    return CPU_ISSET(Id, &Mask);
  }

  return 0;
}
#endif

auto Environment::evaluateCpuAffinity(unsigned RequestedNumThreads, std::string cpuBind) -> int {
#if not((defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY))
  (void)cpuBind;
#endif

  if (RequestedNumThreads > 0 && RequestedNumThreads > this->topology().numThreads()) {
    log::warn() << "Not enough CPUs for requested number of threads";
  }

#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  cpu_set_t Cpuset;

  CPU_ZERO(&Cpuset);

  if (cpuBind.empty()) {
    // no cpu binding defined

    // use all CPUs if not defined otherwise
    if (RequestedNumThreads == 0) {
      for (unsigned I = 0; I < this->topology().maxNumThreads(); I++) {
        if (this->cpuAllowed(I)) {
          CPU_SET(I, &Cpuset);
          RequestedNumThreads++;
        }
      }
    } else {
      // if -n / --threads is set
      unsigned CpuCount = 0;
      for (unsigned I = 0; I < this->topology().maxNumThreads(); I++) {
        // skip if cpu is not available
        if (!this->cpuAllowed(I)) {
          continue;
        }
        ADD_CPU_SET(I, Cpuset);
        CpuCount++;
        // we reached the desired amounts of threads
        if (CpuCount >= RequestedNumThreads) {
          break;
        }
      }
      // requested to many threads
      if (CpuCount < RequestedNumThreads) {
        log::error() << "You are requesting more threads than "
                        "there are CPUs available in the given cpuset.\n"
                     << "This can be caused by the taskset tool, cgrous, "
                        "the batch system, or similar mechanisms.\n"
                     << "Please fix the -n/--threads argument to match the "
                        "restrictions.";
        return EACCES;
      }
    }
  } else {
    // parse CPULIST for binding
    const std::string Delimiter = ",";
    const std::regex Re(R"(^(?:(\d+)(?:-([1-9]\d*)(?:\/([1-9]\d*))?)?)$)");

    std::stringstream Ss(cpuBind);

    while (Ss.good()) {
      std::string Token;
      std::smatch M;
      std::getline(Ss, Token, ',');
      ;

      if (std::regex_match(Token, M, Re)) {
        unsigned long Y;
        unsigned long S;

        unsigned long X = std::stoul(M[1].str());
        if (M[2].matched) {
          Y = std::stoul(M[2].str());
        } else {
          Y = X;
        }
        if (M[3].matched) {
          S = std::stoul(M[3].str());
        } else {
          S = 1;
        }
        if (Y < X) {
          log::error() << "y has to be >= x in x-y expressions of CPU list: " << Token;
          return EXIT_FAILURE;
        }
        for (unsigned long I = X; I <= Y; I += S) {
          ADD_CPU_SET(I, Cpuset);
          RequestedNumThreads++;
        }
      } else {
        log::error() << "Invalid symbols in CPU list: " << Token;
        return EXIT_FAILURE;
      }
    }
  }
#else
  if (requestedNumThreads == 0) {
    requestedNumThreads = this->topology().maxNumThreads();
  }
#endif

  if (RequestedNumThreads == 0) {
    log::error() << "Found no usable CPUs!";
    return 127;
  }
#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  for (unsigned I = 0; I < this->topology().maxNumThreads(); I++) {
    if (CPU_ISSET(I, &Cpuset)) {
      this->CpuBind.push_back(I);
    }
  }

#endif

  RequestedNumThreads = std::min(RequestedNumThreads, this->topology().maxNumThreads());

  this->RequestedNumThreads = RequestedNumThreads;

  return EXIT_SUCCESS;
}

void Environment::printThreadSummary() {
  log::info() << "\n  using " << this->requestedNumThreads() << " threads";

#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  bool PrintCoreIdInfo = false;
  size_t i = 0;

  std::vector<unsigned> CpuBind(this->CpuBind);
  CpuBind.resize(this->requestedNumThreads());
  for (auto const& Bind : CpuBind) {
    int CoreId = this->topology().getCoreIdFromPU(Bind);
    int PkgId = this->topology().getPkgIdFromPU(Bind);

    if (CoreId != -1 && PkgId != -1) {
      log::info() << "    - Thread " << i << " run on CPU " << Bind << ", core " << CoreId << " in package: " << PkgId;
      PrintCoreIdInfo = true;
    }

    i++;
  }

  if (PrintCoreIdInfo) {
    log::info() << "  The cores are numbered using the logical_index from hwloc.";
  }
#endif
}

auto Environment::setCpuAffinity(unsigned Thread) -> int {
  if (Thread >= this->requestedNumThreads()) {
    log::error() << "Trying to set more CPUs than available.";
    return EXIT_FAILURE;
  }

#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
  this->cpuSet(this->CpuBind.at(Thread));
#endif

  return EXIT_SUCCESS;
}

}; // namespace firestarter::environment