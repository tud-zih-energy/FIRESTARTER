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

#pragma once

#include <optional>
#include <ostream>
#include <set>

extern "C" {
#include <hwloc.h>
}

namespace firestarter {

/// This struct describes properties of the threads which are used in the Environment class to assign a specific number
/// of threads and/or use it for cpu binding.
struct HardwareThreadsInfo {
  HardwareThreadsInfo() = default;

  /// The number of hardware threads on this system.
  unsigned MaxNumThreads = 0;
  /// The highest physical index on a hardware thread in the system.
  unsigned MaxPhysicalIndex = 0;
  /// The list of os indices which are available on the system.
  std::set<unsigned> OsIndices;
  /// The optional number of different cpu kinds.
  std::optional<unsigned> CpuKindCount;
};

/// Given that the processor is homogenous. I.e. all cores/packages are of the same type and the are equally
/// distributed, this struct define the numbers of packages, core and threads per core.
struct HomogenousResourceCount {
  /// The number of packages available
  unsigned NumPackagesTotal;
  /// The number of cores available
  unsigned NumCoresTotal;
  /// Assuming we have a consistent number of threads per core. The number of thread per core.
  unsigned NumThreadsPerCore;
};

/// This class models the topology of the processor and its associated packages, cores, threads and caches.
class CPUTopology {
public:
  CPUTopology();
  virtual ~CPUTopology();

  /// Print information about the number of packages, cores and threads.
  void printSystemSummary() const;

  /// Print information about the cache hierarchy.
  void printCacheSummary() const;

  /// Get the size of the first instruction cache.
  [[nodiscard]] auto instructionCacheSize() const -> std::optional<unsigned>;

  /// Given that the processor is homogenous we give back meaningful numbers for the available resouces.
  [[nodiscard]] auto homogenousResourceCount() const -> HomogenousResourceCount;

  /// Get the properties about the hardware threads.
  [[nodiscard]] auto hardwareThreadsInfo() const -> HardwareThreadsInfo;

  /// Get the logical index of the core that housed the PU which is described by the os index.
  /// \arg Pu The os index of the thread.
  /// \returns Optionally the logical index of the CPU that houses this hardware thread.
  [[nodiscard]] auto getCoreIdFromPU(unsigned Pu) const -> std::optional<unsigned>;

  /// Get the logical index of the package that housed the PU which is described by the os index.
  /// \arg Pu The os index of the thread.
  /// \returns Optionally the logical index of the package that houses this hardware thread.
  [[nodiscard]] auto getPkgIdFromPU(unsigned Pu) const -> std::optional<unsigned>;

  /// Set the CPU affinity of the calling thread to the os index in the argument.
  /// \arg OsIndex The os index to which the calling thread should be bound.
  void bindCallerToOsIndex(unsigned OsIndex) const;

private:
  /// The hwloc topology that is used to query information about the processor.
  hwloc_topology_t Topology{};
};

} // namespace firestarter
