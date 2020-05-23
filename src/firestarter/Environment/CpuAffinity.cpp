#include <firestarter/Environment/Environment.hpp>
#include <firestarter/Logging/Log.hpp>

#include <iterator>
#include <regex>
#include <string>

using namespace firestarter::environment;

#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)

extern "C" {
#include <sched.h>
}

// this code is from the C version of FIRESTARTER
// TODO: replace this with cpu affinity of hwloc
#define ADD_CPU_SET(cpu, cpuset)                                               \
  do {                                                                         \
    if (this->cpu_allowed(cpu)) {                                              \
      CPU_SET(cpu, &cpuset);                                                   \
    } else {                                                                   \
      if (cpu >= this->numThreads) {                                           \
        log::error()                                                           \
            << "Error: The given bind argument (-b/--bind) includes CPU "      \
            << cpu << " that is not available on this system.";                \
      } else {                                                                 \
        log::error() << "Error: The given bind argument (-b/--bind) cannot "   \
                        "be implemented with the cpuset given from the OS\n"   \
                     << "This can be caused by the taskset tool, cgroups, "    \
                        "the batch system, or similar mechanisms.\n"           \
                     << "Please fix the argument to match the restrictions.";  \
      }                                                                        \
      return EACCES;                                                           \
    }                                                                          \
  } while (0)

int Environment::cpu_set(unsigned id) {
  cpu_set_t mask;

  CPU_ZERO(&mask);
  CPU_SET(id, &mask);

  return sched_setaffinity(0, sizeof(cpu_set_t), &mask);
}

int Environment::cpu_allowed(unsigned id) {
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

  if (requestedNumThreads > 0 && requestedNumThreads > this->numThreads) {
    log::warn() << "Warning: not enough CPUs for requested number of threads";
  }

#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
  cpu_set_t cpuset;

  CPU_ZERO(&cpuset);

  if (cpuBind.empty()) {
    // no cpu binding defined

    // use all CPUs if not defined otherwise
    if (requestedNumThreads == 0) {
      for (int i = 0; i < this->numThreads; i++) {
        if (this->cpu_allowed(i)) {
          CPU_SET(i, &cpuset);
          requestedNumThreads++;
        }
      }
    } else {
      // if -n / --threads is set
      int current_cpu = 0;
      for (int i = 0; i < this->numThreads; i++) {
        // search for available cpu
        while (!this->cpu_allowed(current_cpu)) {
          current_cpu++;

          // if rearhed end of avail cpus or max(int)
          if (current_cpu >= this->numThreads || current_cpu < 0) {
            log::error() << "Error: Your are requesting more threads than "
                            "there are CPUs available in the given cpuset.\n"
                         << "This can be caused by the taskset tool, cgrous, "
                            "the batch system, or similar mechanisms.\n"
                         << "Please fix the -n/--threads argument to match the "
                            "restrictions.";
            return EACCES;
          }
        }
        ADD_CPU_SET(current_cpu, cpuset);

        // next cpu for next thread (or one of the following)
        current_cpu++;
      }
    }
  } else {
    // parse CPULIST for binding
    const std::string delimiter = ",";
    const std::regex re("^(?:(\\d+)(?:-([1-9]\\d*)(?:\\/([1-9]\\d*))?)?)$");

    size_t pos = 0;

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
          log::error()
              << "Error: y has to be >= x in x-y expressions of CPU list: "
              << token;
          return EXIT_FAILURE;
        }
        for (unsigned long i = x; i <= y; i += s) {
          ADD_CPU_SET(i, cpuset);
          requestedNumThreads++;
        }
      } else {
        log::error() << "Error: invalid symbols in CPU list: " << token;
        return EXIT_FAILURE;
      }
    }
  }
#else
  if (requestedNumThreads == 0) {
    requestedNumThreads = this->numThreads;
  }
#endif

  if (requestedNumThreads == 0) {
    log::error() << "Error: found no usable CPUs!";
    return 127;
  }
#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
  else {
    for (int i = 0; i < this->numThreads; i++) {
      if (CPU_ISSET(i, &cpuset)) {
        this->cpuBind.push_back(i);
      }
    }
  }
#endif

  if (requestedNumThreads > this->numThreads) {
    requestedNumThreads = this->numThreads;
  }

  this->_requestedNumThreads = requestedNumThreads;

  return EXIT_SUCCESS;
}

int Environment::getCoreIdFromPU(unsigned pu) {
  int width;
  hwloc_obj_t obj;

  width = hwloc_get_nbobjs_by_type(this->topology, HWLOC_OBJ_PU);

  if (width >= 1) {
    for (int i = 0; i < width; i++) {
      obj = hwloc_get_obj_by_type(this->topology, HWLOC_OBJ_PU, i);
      if (obj->os_index == pu) {
        for (; obj; obj = obj->parent) {
          if (obj->type == HWLOC_OBJ_CORE) {
            return obj->logical_index;
          }
        }
      }
    }
  }

  return -1;
}

int Environment::getPkgIdFromPU(unsigned pu) {
  int width;
  hwloc_obj_t obj;

  width = hwloc_get_nbobjs_by_type(this->topology, HWLOC_OBJ_PU);

  if (width >= 1) {
    for (int i = 0; i < width; i++) {
      obj = hwloc_get_obj_by_type(this->topology, HWLOC_OBJ_PU, i);
      if (obj->os_index == pu) {
        for (; obj; obj = obj->parent) {
          if (obj->type == HWLOC_OBJ_PACKAGE) {
            return obj->logical_index;
          }
        }
      }
    }
  }

  return -1;
}

void Environment::printThreadSummary(void) {
  log::info() << "\n  using " << this->requestedNumThreads << " threads";

#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
  bool printCoreIdInfo = false;
  size_t i = 0;

  for (auto const &bind : this->cpuBind) {
    int coreId = this->getCoreIdFromPU(bind);
    int pkgId = this->getPkgIdFromPU(bind);

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
  if (thread >= this->requestedNumThreads) {
    log::error() << "Error: Trying to set more CPUs than available.";
    return EXIT_FAILURE;
  }

#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
  this->cpu_set(this->cpuBind.at(thread));
#endif

  return EXIT_SUCCESS;
}
