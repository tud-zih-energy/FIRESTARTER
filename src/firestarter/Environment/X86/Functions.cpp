#include <firestarter/Logging/Log.hpp>
#include <firestarter/Environment/X86/X86Environment.hpp>

#include <firestarter/Environment/X86/Platform/KnightsLandingConfig.hpp>
#include <firestarter/Environment/X86/Platform/SandyBridgeConfig.hpp>
#include <firestarter/Environment/X86/Platform/SandyBridgeEPConfig.hpp>

#include <cstdio>

using namespace firestarter::environment::x86;

void X86Environment::evaluateFunctions(void) {
	for (auto ctor : this->platformConfigsCtor) {
		// add asmjit for model and family detection
		this->platformConfigs.push_back(ctor(&this->cpuFeatures, 6, 86, this->numThreads / this->numPhysicalCoresPerPackage / this->numPackages));
	}
}

int X86Environment::selectFunction(unsigned functionId) {
	unsigned id = 1;

	// if functionId is 0 get the default or fallback

	for (auto config : this->platformConfigs) {
		for (auto const& [thread, functionName] : config->getThreadMap()) {
			// found function
			if (id == functionId) {
				if (!config->isAvailable()) {
					log::error() << "Error: Function " << functionId << "(\"" << functionName << "\") requires ";
					return EXIT_FAILURE;
				}
				return EXIT_SUCCESS;
			}
			id++;
		}
	}

	log::error() << "Error: unknown function id: " << functionId << ", see --avail for available ids";

	return EXIT_FAILURE;
}

void X86Environment::printFunctionSummary() {
	log::info()
		<< " available load-functions:\n"
		<< "  ID   | NAME                           | available on this system\n"
		<< "  ----------------------------------------------------------------";

	unsigned id = 1;

	for (auto const& config : this->platformConfigs) {
		for (auto const& [thread, functionName] : config->getThreadMap()) {
			const char *available = config->isAvailable() ? "yes":"no";
			const char *fmt = "  %4u | %-30s | %s";
			int sz = std::snprintf(nullptr, 0, fmt, id, functionName.c_str(), available);
			std::vector<char> buf(sz + 1);
			std::snprintf(&buf[0], buf.size(), fmt, id, functionName.c_str(), available);
			log::info() << std::string(&buf[0]);
			id++;
		}
	}
}
