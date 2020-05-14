#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_HASWELLEPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_HASWELLEPCONFIG_H

#include <firestarter/Environment/X86/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>

namespace firestarter::environment::x86::platform {
	class HaswellEPConfig : public PlatformConfig {

		public:
			HaswellEPConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				PlatformConfig("HSW_XEONEP", 6, {63,79}, {1,2}, family, model, threads, new payload::FMAPayload(supportedFeatures)) {};

			~HaswellEPConfig() {};
	};
}

#endif
