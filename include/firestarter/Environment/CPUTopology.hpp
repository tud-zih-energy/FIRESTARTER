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
  CPUTopology(std::string architecture);
  virtual ~CPUTopology();

  unsigned numThreads() const { return _numThreadsPerCore * _numCoresTotal; }
  unsigned maxNumThreads() const;
  unsigned numThreadsPerCore() const { return _numThreadsPerCore; }
  unsigned numCoresTotal() const { return _numCoresTotal; }
  unsigned numPackages() const { return _numPackages; }

  std::string const& architecture() const { return _architecture; }
  virtual std::string const& vendor() const { return _vendor; }
  virtual std::string const& processorName() const { return _processorName; }
  virtual std::string const& model() const = 0;

  // get the size of the L1i-cache in bytes
  unsigned instructionCacheSize() const { return _instructionCacheSize; }

  // return the cpu clockrate in Hz
  virtual unsigned long long clockrate() const { return _clockrate; }
  // return the cpu features
  virtual std::list<std::string> const& features() const = 0;

  // get a timestamp
  virtual unsigned long long timestamp() const = 0;

  int getPkgIdFromPU(unsigned pu) const;
  int getCoreIdFromPU(unsigned pu) const;

protected:
  std::string scalingGovernor() const;
  std::ostream& print(std::ostream& stream) const;

private:
  static std::stringstream getFileAsStream(std::string const& filePath);

  unsigned _numThreadsPerCore;
  unsigned _numCoresTotal;
  unsigned _numPackages;
  std::string _architecture;
  std::string _vendor = "";
  std::string _processorName = "";
  unsigned _instructionCacheSize = 0;
  unsigned long long _clockrate = 0;
  hwloc_topology_t topology;
};

} // namespace firestarter::environment
