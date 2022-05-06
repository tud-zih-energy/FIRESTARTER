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
         << "    number of processors: " << this->numPackages() << "\n";
  bool is_hybrid = false;
  for (unsigned package = 0; package < this->numPackages(); package++) {
    if (this->numKindsPerPackage(package)>1)
      is_hybrid = true;
  }
  if (is_hybrid) {
    int threads=0;
    for (unsigned package = 0; package < this->numPackages(); package++) {
      stream << "    package  " << package << ":\n";
      for (unsigned kind = 0; kind < this->numKindsPerPackage(package); kind++) {
        int nr_cores = this->numCoresPerPackage(package, kind);
        int nr_threads = this->numThreadsPerCore(package, kind);
        stream << "      core type: " << kind
               << "\n"
               << "        number of cores: " << nr_cores
               << "\n"
               << "        number of threads per core: " << nr_threads
               << "\n";
        threads+=nr_cores*nr_threads;
      }
    }
    stream << "    total number of threads: " << threads << "\n\n";
  } else {
    stream << "    number of cores per package: " << this->numCoresPerPackage(0,0)
           << "\n"
           << "    number of threads per core:  " << this->numThreadsPerCore(0,0)
           << "\n"
           << "    total number of threads:     " << this->maxNumThreads() << "\n\n";
  }
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
         << "    Caches (per core type):";

  std::vector<hwloc_obj_type_t> caches = {
      HWLOC_OBJ_L1CACHE,  HWLOC_OBJ_L1ICACHE, HWLOC_OBJ_L2CACHE,
      HWLOC_OBJ_L2ICACHE, HWLOC_OBJ_L3CACHE,  HWLOC_OBJ_L3ICACHE,
      HWLOC_OBJ_L4CACHE,  HWLOC_OBJ_L5CACHE,
  };

  std::vector<std::string> cacheStrings = {};

  int cc=-1;
  for (unsigned kind=0;kind<this->numKindsPerPackage(0);kind++) {

    if (is_hybrid) {
      stream << "\nCore type " << kind <<":";
    }

    hwloc_bitmap_t bitmap_kind = hwloc_bitmap_alloc();
    if (bitmap_kind == NULL) {
      stream << "Could not allocate memory for Core bitmap";
      return stream;
    }
    // Get CPU bitmap per kind
    int result = hwloc_cpukinds_get_info(this->topology, kind, bitmap_kind,
                                         NULL, NULL, NULL, 0);
    //
    if (result) {
        stream << "Could not get information for Core type "
                  << kind
                  << "Error: "
                  <<  strerror(errno);
      hwloc_bitmap_free(bitmap_kind);
      return stream;
    } else {
        bool first_common_cache=false;
  for (hwloc_obj_type_t const &cache : caches) {
    char string[128];
    int shared;
    hwloc_obj_t cacheObj;

      std::stringstream ss;
      cacheObj = hwloc_get_next_obj_inside_cpuset_by_type(this->topology, bitmap_kind, cache, NULL);

    if ( (cacheObj == NULL ) && ( kind == ( this->numKindsPerPackage(0) - 1 ) ) )
    {
        cacheObj = hwloc_get_next_obj_by_type(this->topology, cache, NULL);
        if (cacheObj != NULL ) {
            ss << "\n Common Cache:";
            first_common_cache=true;
        }
    }
    if (cacheObj != NULL ) {
      ss << "\n      - ";
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
      int nr_caches_in_set = first_common_cache?
          hwloc_get_nbobjs_by_type(this->topology, cache):
          hwloc_get_nbobjs_inside_cpuset_by_type(this->topology, bitmap_kind, cache);

      int nr_threads_in_set = first_common_cache?
          hwloc_get_nbobjs_by_type(this->topology, HWLOC_OBJ_PU):
          hwloc_get_nbobjs_inside_cpuset_by_type(this->topology, bitmap_kind, HWLOC_OBJ_PU);

      shared = nr_threads_in_set / nr_caches_in_set;

      if (shared > 1) {
        ss << "shared among " << shared << " threads.";
      } else {
        ss << "per thread." ;
      }
      stream << ss.str();
    }
    }

      hwloc_bitmap_free(bitmap_kind);
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
    case -1: log::warn() << "Hybrid core check failed"; break;
    case  0: log::warn() << "Hybrid core check read no information"; break;
    default: log::trace() << "Number of CPU kinds:" << nr_cpukinds;
  }
  if (nr_cpukinds > 1 ) {
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

unsigned CPUTopology::instructionCacheSize(unsigned kind) const {
  hwloc_bitmap_t bitmap_kind = hwloc_bitmap_alloc();
  if (bitmap_kind == NULL) {
    log::error() << "Could not allocate memory for CPU bitmap";
    return 0;
  }
  // Get CPU bitmap per kind
  int result = hwloc_cpukinds_get_info(this->topology, kind, bitmap_kind,
                                       NULL, NULL, NULL, 0);
  //
  if (result){
    log::warn() << "Could not get information for CPU kind "
                << kind
                << "Error: "
                <<  strerror(errno);
    hwloc_bitmap_free(bitmap_kind);
    return 0;
  } else {
      hwloc_obj_t l1i_obj = hwloc_get_next_obj_inside_cpuset_by_type(
          this->topology, bitmap_kind, HWLOC_OBJ_L1ICACHE, NULL);
      hwloc_bitmap_free(bitmap_kind);
      return l1i_obj->attr->cache.size;
  }
}

unsigned CPUTopology::minimalInstructionCacheSize() const {
  unsigned minimal=0xFFFFFFFF;
  for (unsigned pack=0; pack<this->_numPackages;pack++)
    for (unsigned ki=0;ki<this->numKindsPerPackage(pack);ki++){
      unsigned current_size = this->instructionCacheSize(ki);
      if (current_size < minimal)
        minimal=current_size;
    }
  return minimal;
}

unsigned CPUTopology::numThreadsPerCore(unsigned package, unsigned kind) const {

  hwloc_bitmap_t bitmap_kind = hwloc_bitmap_alloc();
  if (bitmap_kind == NULL) {
    log::error() << "Could not allocate memory for CPU bitmap";
    return 1;
  }
  // Get CPU bitmap per kind
  int result = hwloc_cpukinds_get_info(this->topology, kind, bitmap_kind,
                                       NULL, NULL, NULL, 0);
  //
  if (result){
    log::warn() << "Could not get information for CPU kind "
                << kind
                << "Error: "
                <<  strerror(errno);
    hwloc_bitmap_free(bitmap_kind);
    return 1;
  } else {

      // go through packages:
      hwloc_obj_t current_package =
          hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PACKAGE, NULL);
      // TODO: check returnvalue
      // now we are at package 0, maybe go to another package
      for (unsigned i = 0; i< package; i++ ) {
        current_package =
              hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PACKAGE, current_package);
        // TODO: check returnvalue
      }

      if (hwloc_bitmap_and(bitmap_kind, bitmap_kind, current_package->cpuset)) {
          log::warn() << "Could not combine CPUset information";
          hwloc_bitmap_free(bitmap_kind);
          return hwloc_get_nbobjs_inside_cpuset_by_type(topology, bitmap_kind, HWLOC_OBJ_CORE);
      }
    int nr_cores = hwloc_get_nbobjs_inside_cpuset_by_type(topology, bitmap_kind, HWLOC_OBJ_CORE);
    int nr_threads = hwloc_get_nbobjs_inside_cpuset_by_type(topology, bitmap_kind, HWLOC_OBJ_PU);
    hwloc_bitmap_free(bitmap_kind);
    return nr_threads/nr_cores;
  }
}
unsigned CPUTopology::numCoresPerPackage(unsigned package, unsigned kind) const {

  // 1. get cpu - bitmap. to do so, we have to
  // 1.1. get bitmap of the package
  // 1.2. get bitmap for kind
  // 1.3. combine them by AND

  hwloc_bitmap_t bitmap_all = hwloc_bitmap_alloc();
  hwloc_bitmap_t bitmap_kind = hwloc_bitmap_alloc();
  if (bitmap_kind == NULL || bitmap_all == NULL) {
    log::error() << "Could not allocate memory for CPU bitmap";
    return 1;
  }
  // go through packages:
  hwloc_obj_t current_package =
      hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PACKAGE, NULL);
  // TODO: check returnvalue
  // now we are at package 0, maybe go to another package
  for (unsigned i = 0; i< package; i++ ) {
    current_package =
          hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PACKAGE, current_package);
    // TODO: check returnvalue
  }

  // Get CPU bitmap per kind
  int result = hwloc_cpukinds_get_info(this->topology, kind, bitmap_kind,
                                       NULL, NULL, NULL, 0);
  //
  if (result){
    log::warn() << "Could not get information for CPU kind "
                << kind
                << "Error: "
                <<  strerror(errno);
    return hwloc_get_nbobjs_inside_cpuset_by_type(topology, current_package->cpuset, HWLOC_OBJ_CORE);
  } else {
    // returns -1 on error
    if (hwloc_bitmap_and(bitmap_all, bitmap_kind, current_package->cpuset)) {
        log::warn() << "Could not combine CPUset information";
        return hwloc_get_nbobjs_inside_cpuset_by_type(topology, current_package->cpuset, HWLOC_OBJ_CORE);
    }
    // no error
    return hwloc_get_nbobjs_inside_cpuset_by_type(topology, bitmap_all, HWLOC_OBJ_CORE);
  }
}


unsigned CPUTopology::numKindsPerPackage(unsigned package) const {
  // There might be more then one kind of cores
  int nr_cpukinds = hwloc_cpukinds_get_nr(this->topology, package);

  // fallback in case this did not work ... can happen on some platforms
  // already printed a warning earlier
  if (nr_cpukinds < 1)
      return 1;
  else
    return nr_cpukinds;
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
  for (int kind_index = 0; kind_index < nr_cpukinds; kind_index++){
    int result = hwloc_cpukinds_get_info(this->topology, kind_index, bitmap,
                                         NULL, NULL, NULL, 0);
    if (result){
      log::warn() << "Could not get information for CPU kind " << kind_index;
    }
    max += hwloc_bitmap_weight(bitmap);
  }

  hwloc_bitmap_free(bitmap);

  return max;
}
