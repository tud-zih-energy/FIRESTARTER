#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_ENVIRONMENT_HPP
#define INCLUDE_FIRESTARTER_ENVIRONMENT_ENVIRONMENT_HPP

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/MemoryBuffer.h>

#include <list>

extern "C" {
#include <firestarter/Compat/util.h>

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

			// CpuAffinity.cpp
			int evaluateCpuAffinity(unsigned requestedNumThreads, std::string cpuBind);

			virtual void evaluateFunctions(void) =0;
			virtual int selectFunction(unsigned functionId) =0;
			virtual void printFunctionSummary(void) =0;

		protected:
			// CpuClockrate.cpp
			std::unique_ptr<llvm::MemoryBuffer> getFileAsStream(std::string filePath, bool showError = true);
			std::unique_ptr<llvm::MemoryBuffer> getScalingGovernor(void);
			virtual int getCpuClockrate(void);

			// CpuAffinity.cpp
			int parse_cpulist(cpu_set_t *cpuset, const char *fsbind, unsigned *requestedNumThreads);

		public:
			int getCoreIdFromPU(unsigned long long pu);
			int getPkgIdFromPU(unsigned long long pu);
			int cpu_allowed(int id);
			int cpu_set(int id);

		protected:
			virtual std::string getModel(void) {
				return llvm::sys::getHostCPUName().str();
			}

			// Environment.cpp
			hwloc_topology_t topology;
			unsigned int numPackages;
			unsigned int numPhysicalCoresPerPackage;
			unsigned int numThreads;
			std::string architecture = std::string("");
			std::string vendor = std::string("");
			std::string processorName = std::string("");
		public:
			std::string model = std::string("");
		protected:
			unsigned long long clockrate;
		public:
			llvm::StringMap<bool> cpuFeatures;

			// CpuAffinity.cpp
			unsigned requestedNumThreads;
			std::list<unsigned long long> cpuBind;
	};

}

#endif
