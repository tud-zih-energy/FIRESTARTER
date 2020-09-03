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

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Support/Host.h>

#include <regex>
#include <thread>

using namespace firestarter::environment;

Environment::Environment(void) {

  hwloc_topology_init(&this->topology);

  // do not filter icaches
  hwloc_topology_set_cache_types_filter(this->topology,
                                        HWLOC_TYPE_FILTER_KEEP_ALL);

  hwloc_topology_load(this->topology);
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

  for (auto &ent : this->cpuFeatures) {
    if (ent.getValue()) {
      ss << ent.getKey().str() << " ";
    }
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

std::unique_ptr<llvm::MemoryBuffer>
Environment::getFileAsStream(std::string filePath, bool showError) {
  std::error_code e;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileStream =
      llvm::MemoryBuffer::getFileAsStream(filePath);
  if ((e = fileStream.getError()) && showError) {
    firestarter::log::error() << filePath << e.message();
    return nullptr;
  }

  return std::move(*fileStream);
}

int Environment::evaluateEnvironment(void) {

  int depth;

  depth = hwloc_get_type_depth(this->topology, HWLOC_OBJ_PACKAGE);

  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    this->numPackages = 1;
  } else {
    this->numPackages = hwloc_get_nbobjs_by_depth(this->topology, depth);
  }

  this->numPhysicalCoresPerPackage =
      llvm::sys::getHostNumPhysicalCores() / this->numPackages;
  this->numThreads = std::thread::hardware_concurrency();

  llvm::sys::getHostCPUFeatures(this->cpuFeatures);

  this->processorName = this->getProcessorName();
  this->vendor = this->getVendor();

  llvm::Triple PT(llvm::sys::getProcessTriple());

  this->architecture = PT.getArchName().str();
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
  if (nullptr == procCpuinfo) {
    log::warn() << "Could not open /proc/cpuinfo";
    return "";
  }

  std::stringstream ss(procCpuinfo->getBuffer().str());
  std::string line;

  while (std::getline(ss, line, '\n')) {
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
  if (nullptr == procCpuinfo) {
    log::warn() << "Could not open /proc/cpuinfo";
    return "";
  }

  std::stringstream ss(procCpuinfo->getBuffer().str());
  std::string line;

  while (std::getline(ss, line, '\n')) {
    const std::regex vendorIdRe("^vendor_id.*:\\s*(.*)\\s*$");
    std::smatch m;

    if (std::regex_match(line, m, vendorIdRe)) {
      return m[1].str();
    }
  }

  log::warn() << "Could determine vendor from /proc/cpuinfo";
  return "";
}
