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

#include "firestarter/CPUTopology.hpp"
#include "firestarter/Logging/Log.hpp"

#include <array>

#if defined(__APPLE__)
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#endif

namespace firestarter {

CPUTopology::CPUTopology() {
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
}

CPUTopology::~CPUTopology() { hwloc_topology_destroy(Topology); }

void CPUTopology::printSystemSummary(std::ostream& Stream) const {
  auto Resouces = homogenousResourceCount();

  Stream << "  system summary:\n"
         << "    number of processors:        " << Resouces.NumCoresTotal << "\n"
         << "    number of cores (total)):    " << Resouces.NumCoresTotal << "\n"
         << "  (this includes only cores in the cgroup)"
         << "\n"
         << "    number of threads per core:  " << Resouces.NumPackagesTotal << "\n"
         << "    total number of threads:     " << hardwareThreadsInfo().MaxNumThreads << "\n";
}

void CPUTopology::printCacheSummary(std::ostream& Stream) const {
  Stream << "    Caches:";

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

      auto Shared = hardwareThreadsInfo().MaxNumThreads / Width;

      if (Shared > 1) {
        Ss << "shared among " << Shared << " threads.";
      } else {
        Ss << "per thread.";
      }

      Stream << Ss.str();
    }
  }
}

auto CPUTopology::instructionCacheSize() const -> std::optional<unsigned> {
  const auto Width = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_L1ICACHE);

  if (Width >= 1) {
    hwloc_obj_t CacheObj = hwloc_get_obj_by_type(Topology, HWLOC_OBJ_L1ICACHE, 0);
    return CacheObj->attr->cache.size;
  }

  return {};
}

auto CPUTopology::homogenousResourceCount() const -> HomogenousResourceCount {
  HomogenousResourceCount Resouces{};

  // get number of packages
  int Depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_PACKAGE);

  if (Depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    Resouces.NumPackagesTotal = 1;
    log::warn() << "No information on number of packages";
  } else {
    Resouces.NumPackagesTotal = hwloc_get_nbobjs_by_depth(Topology, Depth);
  }

  log::trace() << "Number of Packages:" << Resouces.NumPackagesTotal;

  // get number of cores per package
  Depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_CORE);

  if (Depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    Resouces.NumCoresTotal = 1;
    log::warn() << "No information on number of cores";
  } else {
    Resouces.NumCoresTotal = hwloc_get_nbobjs_by_depth(Topology, Depth);
  }
  log::trace() << "Number of Cores:" << Resouces.NumCoresTotal;

  // get number of threads per core
  Depth = hwloc_get_type_depth(Topology, HWLOC_OBJ_PU);

  if (Depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    Resouces.NumThreadsPerCore = 1;
    log::warn() << "No information on number of threads";
  } else {
    Resouces.NumThreadsPerCore = hwloc_get_nbobjs_by_depth(Topology, Depth) / Resouces.NumCoresTotal;
  }

  return Resouces;
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

auto CPUTopology::hardwareThreadsInfo() const -> HardwareThreadsInfo {
  HardwareThreadsInfo Infos;

  // Get the number of different kinds of CPUs
  const auto NrCpukinds = hwloc_cpukinds_get_nr(Topology, 0);

  if (NrCpukinds < 0) {
    log::fatal() << "flags to hwloc_cpukinds_get_nr is invalid. This is not expected.";
  }

  // No information about the cpukinds found. Go through all PUs and save the biggest os index.
  if (NrCpukinds == 0) {
    auto Width = hwloc_get_nbobjs_by_type(Topology, HWLOC_OBJ_PU);
    Infos.MaxNumThreads = Width;

    for (int I = 0; I < Width; I++) {
      auto* Obj = hwloc_get_obj_by_type(Topology, HWLOC_OBJ_PU, I);
      Infos.MaxPhysicalIndex = (std::max)(Infos.MaxPhysicalIndex, Obj->os_index);
      Infos.OsIndices.emplace(Obj->os_index);
    }

    return Infos;
  }

  // Allocate bitmap to get CPUs later
  hwloc_bitmap_t Bitmap = hwloc_bitmap_alloc();
  if (Bitmap == nullptr) {
    // Error should abort, otherwise return zero.
    log::fatal() << "Could not allocate memory for CPU bitmap";
    return Infos;
  }

  // Go through all cpukinds and save the biggest os index.
  for (int KindIndex = 0; KindIndex < NrCpukinds; KindIndex++) {
    const auto Result = hwloc_cpukinds_get_info(Topology, KindIndex, Bitmap, nullptr, nullptr, nullptr, 0);
    if (Result) {
      log::warn() << "Could not get information for CPU kind " << KindIndex;
    }

    auto Weight = hwloc_bitmap_weight(Bitmap);
    if (Weight < 0) {
      log::fatal() << "bitmap is full or bitmap is not infinitely set";
    }

    auto MaxIndex = hwloc_bitmap_last(Bitmap);
    if (MaxIndex < 0) {
      log::fatal() << "bitmap is full or bitmap is not infinitely set";
    }

    Infos.MaxNumThreads += Weight;
    Infos.MaxPhysicalIndex = (std::max)(Infos.MaxPhysicalIndex, static_cast<unsigned>(MaxIndex));

    {
      unsigned OsIndex{};
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
      hwloc_bitmap_foreach_begin(OsIndex, Bitmap) Infos.OsIndices.emplace(OsIndex);
      hwloc_bitmap_foreach_end();
    }
  }

  hwloc_bitmap_free(Bitmap);

  return Infos;
}

void CPUTopology::bindCallerToOsIndex(unsigned OsIndex) const {
  // Hwloc support thread binding on Linux and Windows, however not on MacOS
#if defined(__APPLE__)
  pthread_t thread = pthread_self();
  thread_port_t mach_thread = pthread_mach_thread_np(thread);
  thread_affinity_policy_data_t policy = {1 << OsIndex};

  auto ReturnCode =
      thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&policy), 1);
#else
  const auto* Obj = hwloc_get_pu_obj_by_os_index(Topology, OsIndex);
  auto ReturnCode = hwloc_set_cpubind(Topology, Obj->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
#endif

  if (ReturnCode != 0) {
    firestarter::log::warn() << "Could not enfoce binding the curren thread to OsIndex: " << OsIndex;
  }
}

}; // namespace firestarter