#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_KNIGHTSLANDINGCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_KNIGHTSLANDINGCONFIG_H

#include <firestarter/Environment/X86/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>

namespace firestarter::environment::x86::platform {
	class KnightsLandingConfig : public PlatformConfig {

		public:
			KnightsLandingConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				PlatformConfig("KNL_XEONPHI", 6, {87}, {4}, family, model, threads, new payload::AVX512Payload(supportedFeatures)) {};

			~KnightsLandingConfig() {};
	};
}

#endif
