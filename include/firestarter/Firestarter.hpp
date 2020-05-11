#ifndef INCLUDE_FIRESTARTER_FIRESTARTER_HPP
#define INCLUDE_FIRESTARTER_FIRESTARTER_HPP

#include <firestarter/ThreadData.hpp>

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/MemoryBuffer.h>

#include <list>

extern "C" {
#include <firestarter/Compat/util.h>

#include <hwloc.h>

#include <pthread.h>
}

namespace firestarter {
	
	class Firestarter {
		public:
			// TODO: bind this to one cpu
			Firestarter(void);
			~Firestarter(void);

			int evaluateEnvironment(void);
			void printEnvironmentSummary(void);
			void run(void);

		private:
			std::unique_ptr<llvm::MemoryBuffer> getFileAsStream(std::string filePath, bool showError = true);
			std::unique_ptr<llvm::MemoryBuffer> getScalingGovernor(void);
			int getCpuClockrate(void);
			int genericGetCpuClockrate(void);

			static void *threadWorker(void *threadData);

			hwloc_topology_t topology;
			unsigned int numPackages;
			unsigned int numPhysicalCoresPerPackage;
			unsigned int numThreads;
			std::string architecture = std::string("");
			std::string vendor = std::string("");
			std::string processorName = std::string("");
			std::string model = std::string("");
			unsigned long long clockrate;
			llvm::StringMap<bool> cpuFeatures;

			pthread_t *threads;
			std::list<ThreadData *> threadData;
	};

}

#endif
