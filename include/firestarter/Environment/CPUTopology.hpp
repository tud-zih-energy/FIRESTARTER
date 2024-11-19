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

#pragma once

#include <cstdint>
#include <list>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>

extern "C" {
#include <hwloc.h>
}

namespace firestarter::environment {

/// This class models the properties of a processor.
class CPUTopology {
public:
  explicit CPUTopology(std::string Architecture);
  virtual ~CPUTopology();

  friend auto operator<<(std::ostream& Stream, CPUTopology const& CpuTopologyRef) -> std::ostream&;

  /// The total number of hardware threads.
  [[nodiscard]] auto numThreads() const -> unsigned { return NumThreadsPerCore * NumCoresTotal; }
  /// The hightest physical index in the hwloc cpuset.
  [[nodiscard]] auto highestPhysicalIndex() const -> unsigned;
  /// Assuming we have a consistent number of threads per core. The number of thread per core.
  [[nodiscard]] auto numThreadsPerCore() const -> unsigned { return NumThreadsPerCore; }
  /// The total number of cores.
  [[nodiscard]] auto numCoresTotal() const -> unsigned { return NumCoresTotal; }
  /// The total number of packages.
  [[nodiscard]] auto numPackages() const -> unsigned { return NumPackages; }
  /// The CPU architecture e.g., x86_64
  [[nodiscard]] auto architecture() const -> std::string const& { return Architecture; }
  /// The CPU vendor i.e., Intel or AMD.
  [[nodiscard]] virtual auto vendor() const -> std::string const& { return Vendor; }
  /// The processor name, this includes the vendor specific name
  [[nodiscard]] virtual auto processorName() const -> std::string const& { return ProcessorName; }
  /// The model of the processor. With X86 this is the the string of Family, Model and Stepping.
  [[nodiscard]] virtual auto model() const -> std::string const& = 0;

  /// Getter for the L1i-cache size in bytes
  [[nodiscard]] auto instructionCacheSize() const -> const auto& { return InstructionCacheSize; }

  /// Getter for the clockrate in Hz
  [[nodiscard]] virtual auto clockrate() const -> uint64_t { return Clockrate; }

  /// Getter for the list of CPU features
  [[nodiscard]] virtual auto features() const -> std::list<std::string> const& = 0;

  /// Get the current hardware timestamp
  [[nodiscard]] virtual auto timestamp() const -> uint64_t = 0;

  /// Get the logical index of the core that housed the PU which is described by the os index.
  /// \arg Pu The os index of the thread.
  /// \returns Optionally the logical index of the CPU that houses this hardware thread.
  [[nodiscard]] auto getCoreIdFromPU(unsigned Pu) const -> std::optional<unsigned>;

  /// Get the logical index of the package that housed the PU which is described by the os index.
  /// \arg Pu The os index of the thread.
  /// \returns Optionally the logical index of the package that houses this hardware thread.
  [[nodiscard]] auto getPkgIdFromPU(unsigned Pu) const -> std::optional<unsigned>;

protected:
  /// Read the scaling_govenor file of cpu0 on linux and return the contents as a string.
  [[nodiscard]] static auto scalingGovernor() -> std::string;

  /// Print the information about this process to a stream.
  [[nodiscard]] auto print(std::ostream& Stream) const -> std::ostream&;

private:
  /// The CPU vendor i.e., Intel or AMD.
  std::string Vendor;

  /// Helper function to open a filepath and return a stringstream with its contents.
  /// \arg FilePath The file to open
  /// \returns A stringstream with the contents of the file.
  [[nodiscard]] static auto getFileAsStream(std::string const& FilePath) -> std::stringstream;

  /// Assuming we have a consistent number of threads per core. The number of thread per core.
  unsigned NumThreadsPerCore;
  /// The total number of cores.
  unsigned NumCoresTotal;
  /// The total number of packages.
  unsigned NumPackages;

  /// The CPU architecture e.g., x86_64
  std::string Architecture;
  /// The processor name, this includes the vendor specific name
  std::string ProcessorName;
  /// The optional size of the instruction cache per core.
  std::optional<unsigned> InstructionCacheSize;
  /// Clockrate of the CPU in Hz
  uint64_t Clockrate = 0;
  /// The hwloc topology that is used to query information about the processor.
  hwloc_topology_t Topology{};
};

inline auto operator<<(std::ostream& Stream, CPUTopology const& CpuTopologyRef) -> std::ostream& {
  return CpuTopologyRef.print(Stream);
}

} // namespace firestarter::environment
