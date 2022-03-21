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

#include <firestarter/Environment/Environment.hpp>
#include <firestarter/Logging/Log.hpp>

#include <iterator>
#include <regex>
#include <string>

using namespace firestarter::environment;

#if (defined(linux) || defined(__linux__)) &&                                  \
    defined(FIRESTARTER_THREAD_AFFINITY)

extern "C" {
#include <sched.h>
}

// this code is from the C version of FIRESTARTER
// TODO: replace this with cpu affinity of hwloc
#define ADD_CPU_SET(cpu, cpuset)                                               \
  do {                                                                         \
    if (this->cpuAllowed(cpu)) {                                               \
      CPU_SET(cpu, &cpuset);                                                   \
    } else {                                                                   \
      if (cpu >= this->topology().numThreads()) {                              \
        log::error() << "The given bind argument (-b/--bind) includes CPU "    \
                     << cpu << " that is not available on this system.";       \
      } else {                                                                 \
        log::error() << "The given bind argument (-b/--bind) cannot "          \
                        "be implemented with the cpuset given from the OS\n"   \
                     << "This can be caused by the taskset tool, cgroups, "    \
                        "the batch system, or similar mechanisms.\n"           \
                     << "Please fix the argument to match the restrictions.";  \
      }                                                                        \
      return EACCES;                                                           \
    }                                                                          \
  } while (0)

int Environment::cpuSet(unsigned id) {
  cpu_set_t mask;

  CPU_ZERO(&mask);
  CPU_SET(id, &mask);

  return sched_setaffinity(0, sizeof(cpu_set_t), &mask);
}

int Environment::cpuAllowed(unsigned id) {
  cpu_set_t mask;

  CPU_ZERO(&mask);

  if (!sched_getaffinity(0, sizeof(cpu_set_t), &mask)) {
    return CPU_ISSET(id, &mask);
  }

  return 0;
}
#endif

int Environment::evaluateCpuAffinity(unsigned requestedNumThreads,
                                     std::string cpuBind) {
#if not((defined(linux) || defined(__linux__)) &&                              \
        defined(FIRESTARTER_THREAD_AFFINITY))
  (void)cpuBind;
#endif

  if (requestedNumThreads > 0 &&
      requestedNumThreads > this->topology().numThreads()) {
    log::warn() << "Not enough CPUs for requested number of threads";
  }

#if (defined(linux) || defined(__linux__)) &&                                  \
    defined(FIRESTARTER_THREAD_AFFINITY)
  cpu_set_t cpuset;

  CPU_ZERO(&cpuset);

  if (cpuBind.empty()) {
    // no cpu binding defined

    // use all CPUs if not defined otherwise
    if (requestedNumThreads == 0) {
      for (unsigned i = 0; i < this->topology().maxNumThreads(); i++) {
        if (this->cpuAllowed(i)) {
          CPU_SET(i, &cpuset);
          requestedNumThreads++;
        }
      }
    } else {
      // if -n / --threads is set
      unsigned cpu_count = 0;
      for (unsigned i = 0; i < this->topology().maxNumThreads(); i++) {
        // skip if cpu is not available
        if (!this->cpuAllowed(i)) {
          continue;
        }
        ADD_CPU_SET(i, cpuset);
        cpu_count++;
        // we reached the desired amounts of threads
        if (cpu_count >= requestedNumThreads) {
          break;
        }
      }
      // requested to many threads
      if (cpu_count < requestedNumThreads) {
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
    const std::string delimiter = ",";
    const std::regex re("^(?:(\\d+)(?:-([1-9]\\d*)(?:\\/([1-9]\\d*))?)?)$");

    std::stringstream ss(cpuBind);

    while (ss.good()) {
      std::string token;
      std::smatch m;
      std::getline(ss, token, ',');
      ;

      if (std::regex_match(token, m, re)) {
        unsigned long x, y, s;

        x = std::stoul(m[1].str());
        if (m[2].matched) {
          y = std::stoul(m[2].str());
        } else {
          y = x;
        }
        if (m[3].matched) {
          s = std::stoul(m[3].str());
        } else {
          s = 1;
        }
        if (y < x) {
          log::error() << "y has to be >= x in x-y expressions of CPU list: "
                       << token;
          return EXIT_FAILURE;
        }
        for (unsigned long i = x; i <= y; i += s) {
          ADD_CPU_SET(i, cpuset);
          requestedNumThreads++;
        }
      } else {
        log::error() << "Invalid symbols in CPU list: " << token;
        return EXIT_FAILURE;
      }
    }
  }
#else
  if (requestedNumThreads == 0) {
    requestedNumThreads = this->topology().maxNumThreads();
  }
#endif

  if (requestedNumThreads == 0) {
    log::error() << "Found no usable CPUs!";
    return 127;
  }
#if (defined(linux) || defined(__linux__)) &&                                  \
    defined(FIRESTARTER_THREAD_AFFINITY)
  else {
    for (unsigned i = 0; i < this->topology().maxNumThreads(); i++) {
      if (CPU_ISSET(i, &cpuset)) {
        this->cpuBind.push_back(i);
      }
    }
  }
#endif

  if (requestedNumThreads > this->topology().maxNumThreads()) {
    requestedNumThreads = this->topology().maxNumThreads();
  }

  this->_requestedNumThreads = requestedNumThreads;

  return EXIT_SUCCESS;
}

void Environment::printThreadSummary() {
  log::info() << "\n  using " << this->requestedNumThreads() << " threads";

#if (defined(linux) || defined(__linux__)) &&                                  \
    defined(FIRESTARTER_THREAD_AFFINITY)
  bool printCoreIdInfo = false;
  size_t i = 0;

  std::vector<unsigned> cpuBind(this->cpuBind);
  cpuBind.resize(this->requestedNumThreads());
  for (auto const &bind : cpuBind) {
    int coreId = this->topology().getCoreIdFromPU(bind);
    int pkgId = this->topology().getPkgIdFromPU(bind);

    if (coreId != -1 && pkgId != -1) {
      log::info() << "    - Thread " << i << " run on CPU " << bind << ", core "
                  << coreId << " in package: " << pkgId;
      printCoreIdInfo = true;
    }

    i++;
  }

  if (printCoreIdInfo) {
    log::info()
        << "  The cores are numbered using the logical_index from hwloc.";
  }
#endif
}

int Environment::setCpuAffinity(unsigned thread) {
  if (thread >= this->requestedNumThreads()) {
    log::error() << "Trying to set more CPUs than available.";
    return EXIT_FAILURE;
  }

#if (defined(linux) || defined(__linux__)) &&                                  \
    defined(FIRESTARTER_THREAD_AFFINITY)
  this->cpuSet(this->cpuBind.at(thread));
#endif

  return EXIT_SUCCESS;
}
