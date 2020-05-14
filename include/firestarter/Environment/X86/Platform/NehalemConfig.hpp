#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_NEHALEMCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_NEHALEMCONFIG_H

#include <firestarter/Environment/X86/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/SSE2Payload.hpp>

namespace firestarter::environment::x86::platform {
	class NehalemConfig : public PlatformConfig {

		public:
			NehalemConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				PlatformConfig("NHM_COREI", 6, {30,37,23}, {1,2}, family, model, threads, new payload::SSE2Payload(supportedFeatures)) {};

			~NehalemConfig() {};
	};
}

#endif
