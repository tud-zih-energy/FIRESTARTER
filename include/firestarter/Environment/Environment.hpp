#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_ENVIRONMENT_HPP
#define INCLUDE_FIRESTARTER_ENVIRONMENT_ENVIRONMENT_HPP

#include <firestarter/Environment/Platform/Config.hpp>
#include <firestarter/Environment/Platform/PlatformConfig.hpp>

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/MemoryBuffer.h>

#include <vector>

extern "C" {
#include <hwloc.h>
}

namespace firestarter::environment {

class Environment {
public:
  // Environment.cpp
  Environment(void);
  ~Environment(void);

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
  virtual int selectFunction(unsigned functionId) = 0;
  virtual void printFunctionSummary(void) = 0;

  platform::Config *const &selectedConfig = _selectedConfig;

  const unsigned long long &requestedNumThreads = _requestedNumThreads;
  const unsigned long long &clockrate = _clockrate;

protected:
  platform::Config *_selectedConfig = nullptr;

  // CpuAffinity.cpp
  unsigned long long _requestedNumThreads;

  // Environment.cpp
  unsigned int numPackages;
  unsigned int numPhysicalCoresPerPackage;
  unsigned int numThreads;
  std::string architecture = std::string("");
  std::string vendor = std::string("");
  std::string processorName = std::string("");
  unsigned long long _clockrate;
  llvm::StringMap<bool> cpuFeatures;

  // CpuClockrate.cpp
  std::unique_ptr<llvm::MemoryBuffer> getScalingGovernor(void);
  virtual int getCpuClockrate(void);

private:
  // CpuClockrate.cpp
  std::unique_ptr<llvm::MemoryBuffer> getFileAsStream(std::string filePath,
                                                      bool showError = true);

  // CpuAffinity.cpp
  // TODO: replace these functions with the builtins one from hwloc
  int getCoreIdFromPU(unsigned pu);
  int getPkgIdFromPU(unsigned pu);
  int cpu_allowed(unsigned id);
  int cpu_set(unsigned id);

  virtual std::string getModel(void) {
    return llvm::sys::getHostCPUName().str();
  }

  // Environment.cpp
  hwloc_topology_t topology;
  std::string model = std::string("");

  // CpuAffinity.cpp
  std::vector<unsigned> cpuBind;
};

} // namespace firestarter::environment

#endif
