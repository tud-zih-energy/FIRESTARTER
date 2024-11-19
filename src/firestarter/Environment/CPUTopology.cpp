/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2024 TU Dresden, Center for Information Services and High
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

#include "firestarter/Environment/CPUTopology.hpp"
#include "firestarter/Logging/Log.hpp"

#include <array>
#include <fstream>
#include <regex>
#include <utility>

namespace firestarter::environment {

auto CPUTopology::print(std::ostream& Stream) const -> std::ostream& {
  Stream << "  system summary:\n"
         << "    number of processors:        " << numPackages() << "\n"
         << "    number of cores (total)):    " << numCoresTotal() << "\n"
         << "  (this includes only cores in the cgroup)"
         << "\n"
         << "    number of threads per core:  " << numThreadsPerCore() << "\n"
         << "    total number of threads:     " << numThreads() << "\n\n";

  std::stringstream Ss;

  for (auto const& Entry : features()) {
    Ss << Entry << " ";
  }

  Stream << "  processor characteristics:\n"
         << "    architecture:       " << architecture() << "\n"
         << "    vendor:             " << vendor() << "\n"
         << "    processor-name:     " << processorName() << "\n"
         << "    model:              " << model() << "\n"
         << "    frequency:          " << clockrate() / 1000000 << " MHz\n"
         << "    supported features: " << Ss.str() << "\n"
         << "    Caches:";

  const std::vector<hwloc_obj_type_t> Caches = {
      HWLOC_OBJ_L1CACHE, HWLOC_OBJ_L1ICACHE, HWLOC_OBJ_L2CACHE, HWLOC_OBJ_L2ICACHE,
      HWLOC_OBJ_L3CACHE, HWLOC_OBJ_L3ICACHE, HWLOC_OBJ_L4CACHE, HWLOC_OBJ_L5CACHE,
  };

  for (hwloc_obj_type_t const& Cache : Caches) {
    std::stringstream Ss;

    auto Width = hwloc_get_nbobjs_by_type(Topology, Cache);

    if (Width >= 1) {
      Ss << "\n      - ";

      auto* CacheObj = hwloc_get_obj_by_type(Topology, Cache, 0);
      std::array<char, 128> String{};
      auto* StringPtr = String.data();
      hwloc_obj_type_snprintf(StringPtr, sizeof(String), CacheObj, 0);

      switch (CacheObj->attr->cache.type) {
      case HWLOC_OBJ_CACHE_DATA:
        Ss << "Level " << CacheObj->attr->cache.depth << " Data";
        break;
      case HWLOC_OBJ_CACHE_INSTRUCTION:
        Ss << "Level " << CacheObj->attr->cache.depth << " Instruction";
        break;
      case HWLOC_OBJ_CACHE_UNIFIED:
      default:
        Ss << "Unified Level " << CacheObj->attr->cache.depth;
        break;
      }

      Ss << " Cache, " << CacheObj->attr->cache.size / 1024 << " KiB, " << CacheObj->attr->cache.linesize
         << " B Cacheline, ";

      switch (CacheObj->attr->cache.associativity) {
      case -1:
        Ss << "full";
        break;
      case 0:
        Ss << "unknown";
        break;
      default:
        Ss << CacheObj->attr->cache.associativity << "-way set";
        break;
      }

      Ss << " associative, ";

      auto Shared = numThreads() / Width;

      if (Shared > 1) {
        Ss << "shared among " << Shared << " threads.";
      } else {
        Ss << "per thread.";
      }

      Stream << Ss.str();
    }
  }

  return Stream;
}

CPUTopology::CPUTopology(std::string Architecture)
    : Architecture(std::move(Architecture)) {

  hwloc_topology_init(&Topology);

  // do not filter icaches
  hwloc_topology_set_cache_types_filter(Topology, HWLOC_TYPE_FILTER_KEEP_ALL);

  hwloc_topology_load(Topology);

  // check for hybrid processor
  const auto NrCpukinds = hwloc_cpukinds_get_nr(Topology, 0);

  switch (NrCpukinds) {
  case -1:
    log::warn() << "Hybrid core check failed";
    break;
  case 0:
    log::warn() << "Hybrid core check read no information";
    break;
  default:
    log::trace() << "Number of CPU kinds:" << NrCpukinds;
  }
  if (NrCpukinds > 1) {
    log::warn() << "FIRESTARTER detected a hybrid CPU set-up";
  }
  // get number of packages
  int Depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_PACKAGE);

  if (Depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    NumPackages = 1;
    log::warn() << "Could not get number of packages";
  } else {
    NumPackages = hwloc_get_nbobjs_by_depth(Topology, Depth);
  }

  log::trace() << "Number of Packages:" << NumPackages;
  // get number of cores per package
  Depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_CORE);

  if (Depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    NumCoresTotal = 1;
    log::warn() << "Could not get number of cores";
  } else {
    NumCoresTotal = hwloc_get_nbobjs_by_depth(Topology, Depth);
    if (NumCoresTotal == 0) {
      log::warn() << "Could not get number of cores";
      NumCoresTotal = 1;
    }
  }
  log::trace() << "Number of Cores:" << NumCoresTotal;

  // get number of threads per core
  Depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_PU);

  if (Depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    NumThreadsPerCore = 1;
    log::warn() << "Could not get number of threads";
  } else {
    NumThreadsPerCore = hwloc_get_nbobjs_by_depth(Topology, Depth) / NumCoresTotal;
    if (NumThreadsPerCore == 0) {
      log::warn() << "Could not get number of threads per core";
      NumThreadsPerCore = 1;
    }
  }

  // get vendor, processor name and clockrate for linux
#if defined(linux) || defined(__linux__)
  {
    auto ProcCpuinfo = getFileAsStream("/proc/cpuinfo");
    std::string Line;
    std::string ClockrateStr = "0";

    while (std::getline(ProcCpuinfo, Line, '\n')) {
      const std::regex VendorIdRe("^vendor_id.*:\\s*(.*)\\s*$");
      const std::regex ModelNameRe("^model name.*:\\s*(.*)\\s*$");
      const std::regex CpuMHzRe("^cpu MHz.*:\\s*(.*)\\s*$");
      std::smatch VendorIdMatch;
      std::smatch ModelNameMatch;
      std::smatch CpuMHzMatch;

      if (std::regex_match(Line, VendorIdMatch, VendorIdRe)) {
        Vendor = VendorIdMatch[1].str();
      }

      if (std::regex_match(Line, ModelNameMatch, ModelNameRe)) {
        ProcessorName = ModelNameMatch[1].str();
      }

      if (std::regex_match(Line, CpuMHzMatch, CpuMHzRe)) {
        ClockrateStr = CpuMHzMatch[1].str();
      }
    }

    if (Vendor.empty()) {
      log::warn() << "Could determine vendor from /proc/cpuinfo";
    }

    if (ProcessorName.empty()) {
      log::warn() << "Could determine processor-name from /proc/cpuinfo";
    }

    if (ClockrateStr == "0") {
      firestarter::log::warn() << "Can't determine clockrate from /proc/cpuinfo";
    } else {
      firestarter::log::trace() << "Clockrate from /proc/cpuinfo is " << ClockrateStr;
      Clockrate = static_cast<uint64_t>(1000000U) * std::stoi(ClockrateStr);
    }

    auto Governor = scalingGovernor();
    if (!Governor.empty()) {

      auto ScalingCurFreq = getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").str();
      auto CpuinfoCurFreq = getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq").str();
      auto ScalingMaxFreq = getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq").str();
      auto CpuinfoMaxFreq = getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").str();

      if (Governor == "performance" || Governor == "powersave") {
        if (ScalingCurFreq.empty()) {
          if (!CpuinfoCurFreq.empty()) {
            ClockrateStr = CpuinfoCurFreq;
          }
        } else {
          ClockrateStr = ScalingCurFreq;
        }
      } else {
        if (ScalingMaxFreq.empty()) {
          if (!CpuinfoMaxFreq.empty()) {
            ClockrateStr = CpuinfoMaxFreq;
          }
        } else {
          ClockrateStr = ScalingMaxFreq;
        }
      }

      Clockrate = static_cast<uint64_t>(1000U) * std::stoi(ClockrateStr);
    }
  }
#endif

  // try to detect processor name for macos
#ifdef __APPLE__
  {
    // use sysctl to detect the name
    std::array<char, 128> Buffer{};
    const auto* Cmd = "sysctl -n machdep.cpu.brand_string";
    std::unique_ptr<FILE, decltype(&pclose)> Pipe(popen(Cmd, "r"), pclose);
    if (!Pipe) {
      log::warn() << "Could not determine processor-name";
    }
    if (fgets(Buffer.data(), Buffer.size(), Pipe.get()) != nullptr) {
      auto Str = std::string(Buffer.data());
      Str.erase(std::remove(Str.begin(), Str.end(), '\n'), Str.end());
      ProcessorName = Str;
    }
  }
#endif

// try to detect processor name for windows
#ifdef _WIN32
  {
    // use wmic
    std::array<char, 128> Buffer{};
    const auto* Cmd = "wmic cpu get name";
    std::unique_ptr<FILE, decltype(&_pclose)> Pipe(_popen(Cmd, "r"), _pclose);
    if (!Pipe) {
      log::warn() << "Could not determine processor-name";
    }
    auto Line = 0;
    while (fgets(Buffer.data(), Buffer.size(), Pipe.get()) != nullptr) {
      if (Line != 1) {
        Line++;
        continue;
      }

      auto Str = std::string(Buffer.data());
      Str.erase(std::remove(Str.begin(), Str.end(), '\n'), Str.end());
      ProcessorName = Str;
    }
  }
#endif

  // get L1i-Cache size
  const auto Width = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_L1ICACHE);

  if (Width >= 1) {
    hwloc_obj_t CacheObj = hwloc_get_obj_by_type(Topology, HWLOC_OBJ_L1ICACHE, 0);
    InstructionCacheSize = CacheObj->attr->cache.size;
  }
}

CPUTopology::~CPUTopology() { hwloc_topology_destroy(Topology); }

auto CPUTopology::getFileAsStream(std::string const& FilePath) -> std::stringstream {
  std::ifstream File(FilePath);
  std::stringstream Ss;

  if (!File.is_open()) {
    log::trace() << "Could not open " << FilePath;
  } else {
    Ss << File.rdbuf();
    File.close();
  }

  return Ss;
}

auto CPUTopology::scalingGovernor() -> std::string {
  return getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").str();
}

auto CPUTopology::getCoreIdFromPU(unsigned Pu) const -> std::optional<unsigned> {
  auto Width = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_PU);

  if (Width >= 1) {
    for (int I = 0; I < Width; I++) {
      auto* Obj = hwloc_get_obj_by_type(Topology, HWLOC_OBJ_PU, I);
      if (Obj->os_index == Pu) {
        for (; Obj; Obj = Obj->parent) {
          if (Obj->type == HWLOC_OBJ_CORE) {
            return Obj->logical_index;
          }
        }
      }
    }
  }

  return {};
}

auto CPUTopology::getPkgIdFromPU(unsigned Pu) const -> std::optional<unsigned> {
  auto Width = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_PU);

  if (Width >= 1) {
    for (int I = 0; I < Width; I++) {
      auto* Obj = hwloc_get_obj_by_type(Topology, HWLOC_OBJ_PU, I);
      if (Obj->os_index == Pu) {
        for (; Obj; Obj = Obj->parent) {
          if (Obj->type == HWLOC_OBJ_PACKAGE) {
            return Obj->logical_index;
          }
        }
      }
    }
  }

  return {};
}

auto CPUTopology::highestPhysicalIndex() const -> unsigned {
  unsigned Max = 0;

  // There might be more then one kind of cores
  const auto NrCpukinds = hwloc_cpukinds_get_nr(Topology, 0);

  // fallback in case this did not work ... can happen on some platforms
  // already printed a warning earlier
  if (NrCpukinds < 1) {
    auto Width = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_PU);
    unsigned Max = 0;

    for (int I = 0; I < Width; I++) {
      auto* Obj = hwloc_get_obj_by_type(Topology, HWLOC_OBJ_PU, I);
      Max = (std::max)(Max, Obj->os_index);
    }

    return Max + 1;
  }

  // Allocate bitmap to get CPUs later
  hwloc_bitmap_t Bitmap = hwloc_bitmap_alloc();
  if (Bitmap == nullptr) {
    log::error() << "Could not allocate memory for CPU bitmap";
    return 1;
  }

  // Find CPUs per kind
  for (int KindIndex = 0; KindIndex < NrCpukinds; KindIndex++) {
    const auto Result = hwloc_cpukinds_get_info(Topology, KindIndex, Bitmap, nullptr, nullptr, nullptr, 0);
    if (Result) {
      log::warn() << "Could not get information for CPU kind " << KindIndex;
    }
    Max += hwloc_bitmap_last(Bitmap);
  }

  hwloc_bitmap_free(Bitmap);

  return Max;
}

}; // namespace firestarter::environment