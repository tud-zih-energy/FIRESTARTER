#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGEEPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGEEPCONFIG_H

#include <firestarter/Environment/X86/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/AVXPayload.hpp>

namespace firestarter::environment::x86::platform {
	class SandyBridgeEPConfig : public PlatformConfig {

		public:
			SandyBridgeEPConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				PlatformConfig("SNB_XEONEP", 6, {45,62}, {1,2}, family, model, threads, new payload::AVXPayload(supportedFeatures)) {};

			~SandyBridgeEPConfig() {};
	};
}

#endif
