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

#include <firestarter/Environment/CPUTopology.hpp>
#include <firestarter/Logging/Log.hpp>

#include <array>
#include <fstream>
#include <regex>

extern "C" {
#include <stdio.h>
}

using namespace firestarter::environment;

std::ostream &CPUTopology::print(std::ostream &stream) const {
  stream << "  system summary:\n"
         << "    number of processors:        " << this->numPackages() << "\n"
         << "    number of cores per package: " << this->numCoresPerPackage()
         << "\n"
         << "    number of threads per core:  " << this->numThreadsPerCore()
         << "\n"
         << "    total number of threads:     " << this->numThreads() << "\n\n";

  std::stringstream ss;

  for (auto const &ent : this->features()) {
    ss << ent << " ";
  }

  stream << "  processor characteristics:\n"
         << "    architecture:       " << this->architecture() << "\n"
         << "    vendor:             " << this->vendor() << "\n"
         << "    processor-name:     " << this->processorName() << "\n"
         << "    model:              " << this->model() << "\n"
         << "    frequency:          " << this->clockrate() / 1000000
         << " MHz\n"
         << "    supported features: " << ss.str() << "\n"
         << "    Caches:";

  std::vector<hwloc_obj_type_t> caches = {
      HWLOC_OBJ_L1CACHE,  HWLOC_OBJ_L1ICACHE, HWLOC_OBJ_L2CACHE,
      HWLOC_OBJ_L2ICACHE, HWLOC_OBJ_L3CACHE,  HWLOC_OBJ_L3ICACHE,
      HWLOC_OBJ_L4CACHE,  HWLOC_OBJ_L5CACHE,
  };

  std::vector<std::string> cacheStrings = {};

  for (hwloc_obj_type_t const &cache : caches) {
    int width;
    char string[128];
    int shared;
    hwloc_obj_t cacheObj;
    std::stringstream ss;

    width = hwloc_get_nbobjs_by_type(this->topology, cache);

    if (width >= 1) {
      ss << "\n      - ";

      cacheObj = hwloc_get_obj_by_type(this->topology, cache, 0);
      hwloc_obj_type_snprintf(string, sizeof(string), cacheObj, 0);

      switch (cacheObj->attr->cache.type) {
      case HWLOC_OBJ_CACHE_DATA:
        ss << "Level " << cacheObj->attr->cache.depth << " Data";
        break;
      case HWLOC_OBJ_CACHE_INSTRUCTION:
        ss << "Level " << cacheObj->attr->cache.depth << " Instruction";
        break;
      case HWLOC_OBJ_CACHE_UNIFIED:
      default:
        ss << "Unified Level " << cacheObj->attr->cache.depth;
        break;
      }

      ss << " Cache, " << cacheObj->attr->cache.size / 1024 << " KiB, "
         << cacheObj->attr->cache.linesize << " B Cacheline, ";

      switch (cacheObj->attr->cache.associativity) {
      case -1:
        ss << "full";
        break;
      case 0:
        ss << "unknown";
        break;
      default:
        ss << cacheObj->attr->cache.associativity << "-way set";
        break;
      }

      ss << " associative, ";

      shared = this->numThreads() / width;

      if (shared > 1) {
        ss << "shared among " << shared << " threads.";
      } else {
        ss << "per thread.";
      }

      stream << ss.str();
    }
  }

  return stream;
}

CPUTopology::CPUTopology(std::string architecture)
    : _architecture(architecture) {

  hwloc_topology_init(&this->topology);

  // do not filter icaches
  hwloc_topology_set_cache_types_filter(this->topology,
                                        HWLOC_TYPE_FILTER_KEEP_ALL);

  hwloc_topology_load(this->topology);

  // check for hybrid processor
  int nr_cpukinds = hwloc_cpukinds_get_nr(this->topology, 0);

  switch (nr_cpukinds) {
  case -1:
    log::warn() << "Hybrid core check failed";
    break;
  case 0:
    log::warn() << "Hybrid core check read no information";
    break;
  default:
    log::trace() << "Number of CPU kinds:" << nr_cpukinds;
  }
  if (nr_cpukinds > 1) {
    log::warn() << "FIRESTARTER detected a hybrid CPU set-up";
  }

  // get number of packages
  int depth = hwloc_get_type_depth(this->topology, HWLOC_OBJ_PACKAGE);

  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    this->_numPackages = 1;
    log::warn() << "Could not get number of packages";
  } else {
    this->_numPackages = hwloc_get_nbobjs_by_depth(this->topology, depth);
  }

  // get number of cores per package
  depth = hwloc_get_type_depth(this->topology, HWLOC_OBJ_CORE);

  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    this->_numCoresPerPackage = 1;
    log::warn() << "Could not get number of cores";
  } else {
    this->_numCoresPerPackage =
        hwloc_get_nbobjs_by_depth(this->topology, depth) / this->_numPackages;
  }

  // get number of threads per core
  depth = hwloc_get_type_depth(this->topology, HWLOC_OBJ_PU);

  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    this->_numThreadsPerCore = 1;
    log::warn() << "Could not get number of threads";
  } else {
    this->_numThreadsPerCore =
        hwloc_get_nbobjs_by_depth(this->topology, depth) /
        this->_numCoresPerPackage / this->_numPackages;
  }

  // get vendor, processor name and clockrate for linux
#if defined(linux) || defined(__linux__)
  auto procCpuinfo = this->getFileAsStream("/proc/cpuinfo");
  std::string line;
  std::string clockrate = "0";

  while (std::getline(procCpuinfo, line, '\n')) {
    const std::regex vendorIdRe("^vendor_id.*:\\s*(.*)\\s*$");
    const std::regex modelNameRe("^model name.*:\\s*(.*)\\s*$");
    const std::regex cpuMHzRe("^cpu MHz.*:\\s*(.*)\\s*$");
    std::smatch vendorIdM;
    std::smatch modelNameM;
    std::smatch cpuMHzM;

    if (std::regex_match(line, vendorIdM, vendorIdRe)) {
      this->_vendor = vendorIdM[1].str();
    }

    if (std::regex_match(line, modelNameM, modelNameRe)) {
      this->_processorName = modelNameM[1].str();
    }

    if (std::regex_match(line, cpuMHzM, cpuMHzRe)) {
      clockrate = cpuMHzM[1].str();
    }
  }

  if (this->_vendor == "") {
    log::warn() << "Could determine vendor from /proc/cpuinfo";
  }

  if (this->_processorName == "") {
    log::warn() << "Could determine processor-name from /proc/cpuinfo";
  }

  if (clockrate == "0") {
    firestarter::log::warn() << "Can't determine clockrate from /proc/cpuinfo";
  } else {
    firestarter::log::trace()
        << "Clockrate from /proc/cpuinfo is " << clockrate;
    this->_clockrate = 1e6 * std::stoi(clockrate);
  }

  auto governor = this->scalingGovernor();
  if (!governor.empty()) {

    auto scalingCurFreq =
        this->getFileAsStream(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            .str();
    auto cpuinfoCurFreq =
        this->getFileAsStream(
                "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq")
            .str();
    auto scalingMaxFreq =
        this->getFileAsStream(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
            .str();
    auto cpuinfoMaxFreq =
        this->getFileAsStream(
                "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
            .str();

    if (governor.compare("performance") || governor.compare("powersave")) {
      if (scalingCurFreq.empty()) {
        if (!cpuinfoCurFreq.empty()) {
          clockrate = cpuinfoCurFreq;
        }
      } else {
        clockrate = scalingCurFreq;
      }
    } else {
      if (scalingMaxFreq.empty()) {
        if (!cpuinfoMaxFreq.empty()) {
          clockrate = cpuinfoMaxFreq;
        }
      } else {
        clockrate = scalingMaxFreq;
      }
    }

    this->_clockrate = 1e3 * std::stoi(clockrate);
  }
#endif

  // try to detect processor name for macos
#ifdef __APPLE__
  // use sysctl to detect the name
  std::array<char, 128> buffer;
  auto cmd = "sysctl -n machdep.cpu.brand_string";
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    log::warn() << "Could not determine processor-name";
  }
  if (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    auto str = std::string(buffer.data());
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    this->_processorName = str;
  }
#endif

// try to detect processor name for windows
#ifdef _WIN32
  // use wmic
  std::array<char, 128> buffer;
  auto cmd = "wmic cpu get name";
  std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
  if (!pipe) {
    log::warn() << "Could not determine processor-name";
  }
  auto line = 0;
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    if (line != 1) {
      line++;
      continue;
    }

    auto str = std::string(buffer.data());
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    this->_processorName = str;
  }
#endif

  // get L1i-Cache size
  int width = hwloc_get_nbobjs_by_type(this->topology, HWLOC_OBJ_L1ICACHE);

  if (width >= 1) {
    hwloc_obj_t cacheObj =
        hwloc_get_obj_by_type(this->topology, HWLOC_OBJ_L1ICACHE, 0);
    this->_instructionCacheSize = cacheObj->attr->cache.size;
  }
}

CPUTopology::~CPUTopology() { hwloc_topology_destroy(this->topology); }

std::stringstream CPUTopology::getFileAsStream(std::string const &filePath) {
  std::ifstream file(filePath);
  std::stringstream ss;

  if (!file.is_open()) {
    log::trace() << "Could not open " << filePath;
  } else {
    ss << file.rdbuf();
    file.close();
  }

  return ss;
}

std::string CPUTopology::scalingGovernor() const {
  return this
      ->getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
      .str();
}

int CPUTopology::getCoreIdFromPU(unsigned pu) const {
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

int CPUTopology::getPkgIdFromPU(unsigned pu) const {
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

unsigned CPUTopology::maxNumThreads() const {
  unsigned max = 0;

  // There might be more then one kind of cores
  int nr_cpukinds = hwloc_cpukinds_get_nr(this->topology, 0);

  // fallback in case this did not work ... can happen on some platforms
  // already printed a warning earlier
  if (nr_cpukinds < 1) {
    hwloc_obj_t obj;
    int width = hwloc_get_nbobjs_by_type(this->topology, HWLOC_OBJ_PU);
    unsigned max = 0;

    for (int i = 0; i < width; i++) {
      obj = hwloc_get_obj_by_type(this->topology, HWLOC_OBJ_PU, i);
      max = max < obj->os_index ? obj->os_index : max;
    }

    return max + 1;
  }

  // Allocate bitmap to get CPUs later
  hwloc_bitmap_t bitmap = hwloc_bitmap_alloc();
  if (bitmap == NULL) {
    log::error() << "Could not allocate memory for CPU bitmap";
    return 1;
  }

  // Find CPUs per kind
  for (int kind_index = 0; kind_index < nr_cpukinds; kind_index++) {
    int result = hwloc_cpukinds_get_info(this->topology, kind_index, bitmap,
                                         NULL, NULL, NULL, 0);
    if (result) {
      log::warn() << "Could not get information for CPU kind " << kind_index;
    }
    max += hwloc_bitmap_weight(bitmap);
  }

  hwloc_bitmap_free(bitmap);

  return max;
}
