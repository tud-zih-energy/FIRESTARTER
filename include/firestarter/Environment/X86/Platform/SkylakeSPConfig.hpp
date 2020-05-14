#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKESPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKESPCONFIG_H

#include <firestarter/Environment/X86/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>

namespace firestarter::environment::x86::platform {
	class SkylakeSPConfig : public PlatformConfig {

		public:
			SkylakeSPConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				PlatformConfig("SKL_XEONEP", 6, {85}, {1,2}, family, model, threads, new payload::AVX512Payload(supportedFeatures)) {};

			~SkylakeSPConfig() {};
	};
}

#endif
