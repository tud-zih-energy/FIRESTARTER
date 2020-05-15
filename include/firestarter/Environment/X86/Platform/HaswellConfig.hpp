#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_HASWELLCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_HASWELLCONFIG_H

#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>

namespace firestarter::environment::x86::platform {
	class HaswellConfig : public X86PlatformConfig {

		public:
			HaswellConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				X86PlatformConfig("HSW_COREI", 6, {60,61,69,70,71}, {1,2}, family, model, threads, new payload::FMAPayload(supportedFeatures)) {};

			~HaswellConfig() {};
	};
}

#endif
