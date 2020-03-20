#ifndef INCLUDE_FIRESTARTER_FIRESTARTER_HPP
#define INCLUDE_FIRESTARTER_FIRESTARTER_HPP

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/MemoryBuffer.h>

extern "C" {
	#include <firestarter/util.h>
}

namespace firestarter {
	
	class Firestarter {
		public:
			// TODO: bind this to one cpu
			Firestarter(void) {};
			~Firestarter(void) {};

			int evaluateEnvironment(void);
			void printEnvironmentSummary(void);

		private:
			std::unique_ptr<llvm::MemoryBuffer> getFileAsStream(std::string filePath, bool showError = true);
			std::unique_ptr<llvm::MemoryBuffer> getScalingGovernor(void);
			int getCpuClockrate(void);
			int genericGetCpuClockrate(void);

#ifdef __ARCH_X86
			int hasInvariantRdtsc(void);
#endif

			unsigned int numPhysicalCores;
			unsigned int numThreads;
			std::string architecture = std::string("");
			std::string vendor = std::string("");
			std::string processorName = std::string("");
			std::string model = std::string("");
			unsigned long long clockrate;
			llvm::StringMap<bool> cpuFeatures;
	};

}

#endif
