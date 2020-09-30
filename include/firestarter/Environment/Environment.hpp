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

#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_ENVIRONMENT_HPP
#define INCLUDE_FIRESTARTER_ENVIRONMENT_ENVIRONMENT_HPP

#include <firestarter/Environment/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/Platform/RuntimeConfig.hpp>

#include <vector>

extern "C" {
#include <hwloc.h>
}

namespace firestarter::environment {

class Environment {
public:
  // Environment.cpp
  Environment(std::string architecture);
  virtual ~Environment(void);

  // Environment.cpp
  int evaluateEnvironment(void);
  void printEnvironmentSummary(void);
  unsigned getNumberOfThreadsPerCore(void);

  // CpuAffinity.cpp
  int evaluateCpuAffinity(unsigned requestedNumThreads, std::string cpuBind);
  int setCpuAffinity(unsigned thread);
  void printThreadSummary(void);

  virtual unsigned long long timestamp(void) = 0;

  virtual void evaluateFunctions(void) = 0;
  virtual int selectFunction(unsigned functionId,
                             bool allowUnavailablePayload) = 0;
  virtual int selectInstructionGroups(std::string groups) = 0;
  virtual void printAvailableInstructionGroups(void) = 0;
  virtual void printSelectedCodePathSummary(void) = 0;
  virtual void printFunctionSummary(void) = 0;

  platform::RuntimeConfig *const &selectedConfig = _selectedConfig;

  const unsigned long long &requestedNumThreads = _requestedNumThreads;
  const unsigned long long &clockrate = _clockrate;

protected:
  platform::RuntimeConfig *_selectedConfig = nullptr;

  // CpuAffinity.cpp
  unsigned long long _requestedNumThreads;

  // Environment.cpp
  unsigned int numPackages;
  unsigned int numPhysicalCoresPerPackage;
  unsigned int numThreads;
  std::string architecture;
  std::string vendor = std::string("");
  std::string processorName = std::string("");
  unsigned long long _clockrate = 0;
  unsigned instructionCacheSize = 0;

  // CpuClockrate.cpp
  std::stringstream getScalingGovernor(void);
  virtual int getCpuClockrate(void);

  virtual std::string getModel(void) = 0;
  virtual std::string getProcessorName(void);
  virtual std::string getVendor(void);

private:
  // CpuAffinity.cpp
  // TODO: replace these functions with the builtins one from hwloc
  int getCoreIdFromPU(unsigned pu);
  int getPkgIdFromPU(unsigned pu);
  int cpu_allowed(unsigned id);
  int cpu_set(unsigned id);

  // Environment.cpp
  hwloc_topology_t topology;
  std::string model = std::string("");
  std::stringstream getFileAsStream(std::string filePath);

  // CpuAffinity.cpp
  std::vector<unsigned> cpuBind;

  virtual std::list<std::string> getCpuFeatures(void) = 0;
};

} // namespace firestarter::environment

#endif
