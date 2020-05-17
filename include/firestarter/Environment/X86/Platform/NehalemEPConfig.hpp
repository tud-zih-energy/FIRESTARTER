#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_NEHALEMEPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_NEHALEMEPCONFIG_H

#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/SSE2Payload.hpp>

namespace firestarter::environment::x86::platform {
	class NehalemEPConfig : public X86PlatformConfig {

		public:
			NehalemEPConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) :
				X86PlatformConfig("NHM_XEONEP", 6, {26,44}, {1,2},
						{32768,262144,2097152}, 104857600,
						family, model, threads, new payload::SSE2Payload(supportedFeatures)) {};
			~NehalemEPConfig() {};

			std::map<std::string, unsigned> getDefaultPayloadSettings(void) override {
				return std::map<std::string, unsigned>({
					{"RAM_P", 1},
					{"L1_LS", 60},
					{"REG", 2}
				});
			}
	};
}

#endif
