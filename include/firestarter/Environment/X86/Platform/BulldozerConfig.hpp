#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_BULLDOZERCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_BULLDOZERCONFIG_H

#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/FMA4Payload.hpp>

namespace firestarter::environment::x86::platform {
	class BulldozerConfig : public X86PlatformConfig {

		public:
			BulldozerConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				X86PlatformConfig("BLD_OPTERON", 21, {1,2,3}, {1}, family, model, threads, new payload::FMA4Payload(supportedFeatures)) {};

			~BulldozerConfig() {};
	};
}

#endif
