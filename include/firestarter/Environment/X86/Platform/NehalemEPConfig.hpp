#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_NEHALEMEPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_NEHALEMEPCONFIG_H

#include <firestarter/Environment/X86/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/SSE2Payload.hpp>

namespace firestarter::environment::x86::platform {
	class NehalemEPConfig : public PlatformConfig {

		public:
			NehalemEPConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				PlatformConfig("NHM_XEONEP", 6, {26,44}, {1,2}, family, model, threads, new payload::SSE2Payload(supportedFeatures)) {};

			~NehalemEPConfig() {};
	};
}

#endif
