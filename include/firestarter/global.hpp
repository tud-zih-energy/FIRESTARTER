#ifndef INCLUDE_FIRESTARTER_GLOBAL_HPP
#define INCLUDE_FIRESTARTER_GLOBAL_HPP

#include <llvm/ADT/StringMap.h>

namespace firestarter {

	typedef struct { 
		unsigned int numPhysicalCores;
		unsigned int numThreads;
		std::string architecture = std::string("");
		std::string vendor = std::string("");
		std::string processorName = std::string("");
		std::string model = std::string("");
		unsigned long long clockrate;
		llvm::StringMap<bool> cpuFeatures;
	} data_t;

}

#endif
