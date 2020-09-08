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

#include <fstream>
#include <regex>
#include <thread>

using namespace firestarter::environment;

Environment::Environment(std::string architecture) {

  hwloc_topology_init(&this->topology);

  // do not filter icaches
  hwloc_topology_set_cache_types_filter(this->topology,
                                        HWLOC_TYPE_FILTER_KEEP_ALL);

  hwloc_topology_load(this->topology);

  this->architecture = architecture;
}

Environment::~Environment(void) { hwloc_topology_destroy(this->topology); }

void Environment::printEnvironmentSummary(void) {

  log::info() << "  system summary:\n"
              << "    number of processors:        " << this->numPackages
              << "\n"
              << "    number of cores per package: "
              << this->numPhysicalCoresPerPackage << "\n"
              << "    number of threads per core:  "
              << this->numThreads / this->numPhysicalCoresPerPackage /
                     this->numPackages
              << "\n"
              << "    total number of threads:     " << this->numThreads
              << "\n";

  std::stringstream ss;

  for (auto &ent : this->getCpuFeatures()) {
    ss << ent << " ";
  }

  log::info() << "  processor characteristics:\n"
              << "    architecture:       " << this->architecture << "\n"
              << "    vendor:             " << this->vendor << "\n"
              << "    processor-name:     " << this->processorName << "\n"
              << "    model:              " << this->model << "\n"
              << "    frequency:          " << this->clockrate / 1000000
              << " MHz\n"
              << "    supported features: " << ss.str() << "\n"
              << "    Caches:";

  std::vector<hwloc_obj_type_t> caches = {
      HWLOC_OBJ_L1CACHE,  HWLOC_OBJ_L1ICACHE, HWLOC_OBJ_L2CACHE,
      HWLOC_OBJ_L2ICACHE, HWLOC_OBJ_L3CACHE,  HWLOC_OBJ_L3ICACHE,
      HWLOC_OBJ_L4CACHE,  HWLOC_OBJ_L5CACHE,
  };

  std::for_each(
      std::begin(caches), std::end(caches),
      [this](hwloc_obj_type_t const &cache) {
        int width;
        char string[128];
        int shared;
        hwloc_obj_t cacheObj;
        std::stringstream ss;

        width = hwloc_get_nbobjs_by_type(this->topology, cache);

        if (width >= 1) {
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

          shared = this->numThreads / width;

          if (shared > 1) {
            ss << "shared among " << shared << " threads.";
          } else {
            ss << "per thread.";
          }

          log::info() << "      - " << ss.str();
        }
      });
}

std::stringstream Environment::getFileAsStream(std::string filePath) {
  std::ifstream file(filePath);
  std::stringstream ss;

  if (!file.is_open()) {
    log::error() << "Could not open " << filePath;
  } else {
    ss << file.rdbuf();
    file.close();
  }

  return ss;
}

int Environment::evaluateEnvironment(void) {

  int depth;

  depth = hwloc_get_type_depth(this->topology, HWLOC_OBJ_PACKAGE);

  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    this->numPackages = 1;
    log::warn() << "Cound not get number of packages";
  } else {
    this->numPackages = hwloc_get_nbobjs_by_depth(this->topology, depth);
  }

  depth = hwloc_get_type_depth(this->topology, HWLOC_OBJ_CORE);

  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    this->numPhysicalCoresPerPackage = 1;
    log::warn() << "Cound not get number of cores";
  } else {
    this->numPhysicalCoresPerPackage =
        hwloc_get_nbobjs_by_depth(this->topology, depth) / this->numPackages;
  }

  this->numThreads = std::thread::hardware_concurrency();

  this->processorName = this->getProcessorName();
  this->vendor = this->getVendor();

  this->model = this->getModel();

  if (EXIT_SUCCESS != this->getCpuClockrate()) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

unsigned Environment::getNumberOfThreadsPerCore(void) {
  return this->numThreads / this->numPhysicalCoresPerPackage /
         this->numPackages;
}

std::string Environment::getProcessorName(void) {
  auto procCpuinfo = this->getFileAsStream("/proc/cpuinfo");
  if (procCpuinfo.str().empty()) {
    return "";
  }

  std::string line;

  while (std::getline(procCpuinfo, line, '\n')) {
    const std::regex modelNameRe("^model name.*:\\s*(.*)\\s*$");
    std::smatch m;

    if (std::regex_match(line, m, modelNameRe)) {
      return m[1].str();
    }
  }

  log::warn() << "Could determine processor-name from /proc/cpuinfo";
  return "";
}

std::string Environment::getVendor(void) {
  auto procCpuinfo = this->getFileAsStream("/proc/cpuinfo");
  if (procCpuinfo.str().empty()) {
    return "";
  }

  std::string line;

  while (std::getline(procCpuinfo, line, '\n')) {
    const std::regex vendorIdRe("^vendor_id.*:\\s*(.*)\\s*$");
    std::smatch m;

    if (std::regex_match(line, m, vendorIdRe)) {
      return m[1].str();
    }
  }

  log::warn() << "Could determine vendor from /proc/cpuinfo";
  return "";
}
