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

#include <list>
#include <ostream>
#include <sstream>
#include <string>

extern "C" {
#include <hwloc.h>
}

namespace firestarter::environment {

class CPUTopology {
public:
  explicit CPUTopology(std::string Architecture);
  virtual ~CPUTopology();

  [[nodiscard]] auto numThreads() const -> unsigned { return NumThreadsPerCore * NumCoresTotal; }
  [[nodiscard]] auto maxNumThreads() const -> unsigned;
  [[nodiscard]] auto numThreadsPerCore() const -> unsigned { return NumThreadsPerCore; }
  [[nodiscard]] auto numCoresTotal() const -> unsigned { return NumCoresTotal; }
  [[nodiscard]] auto numPackages() const -> unsigned { return NumPackages; }

  [[nodiscard]] auto architecture() const -> std::string const& { return Architecture; }
  [[nodiscard]] virtual auto vendor() const -> std::string const& { return Vendor; }
  [[nodiscard]] virtual auto processorName() const -> std::string const& { return ProcessorName; }
  [[nodiscard]] virtual auto model() const -> std::string const& { return Model; }

  // get the size of the L1i-cache in bytes
  [[nodiscard]] auto instructionCacheSize() const -> unsigned { return InstructionCacheSize; }

  // return the cpu clockrate in Hz
  [[nodiscard]] virtual auto clockrate() const -> uint64_t { return Clockrate; }
  // return the cpu features
  [[nodiscard]] virtual auto features() const -> std::list<std::string> const& = 0;

  // get a timestamp
  [[nodiscard]] virtual auto timestamp() const -> uint64_t = 0;

  [[nodiscard]] auto getPkgIdFromPU(unsigned Pu) const -> int;
  [[nodiscard]] auto getCoreIdFromPU(unsigned Pu) const -> int;

protected:
  [[nodiscard]] static auto scalingGovernor() -> std::string;
  [[nodiscard]] auto print(std::ostream& Stream) const -> std::ostream&;

  std::string Vendor;
  std::string Model;

private:
  [[nodiscard]] static auto getFileAsStream(std::string const& FilePath) -> std::stringstream;

  unsigned NumThreadsPerCore;
  unsigned NumCoresTotal;
  unsigned NumPackages;
  std::string Architecture;
  std::string ProcessorName;
  unsigned InstructionCacheSize = 0;
  uint64_t Clockrate = 0;
  hwloc_topology_t Topology;
};

} // namespace firestarter::environment
