#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGECONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGECONFIG_H

#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/AVXPayload.hpp>

namespace firestarter::environment::x86::platform {
	class SandyBridgeConfig : public X86PlatformConfig {

		public:
			SandyBridgeConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				X86PlatformConfig("SNB_COREI", 6, {42,58}, {1,2}, family, model, threads, new payload::AVXPayload(supportedFeatures)) {};

			~SandyBridgeConfig() {};
	};
}

#endif
