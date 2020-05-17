#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMA4PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMA4PAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {

	class FMA4Payload : public X86Payload {
		public:
			FMA4Payload(llvm::StringMap<bool> *supportedFeatures) : X86Payload(supportedFeatures, {"avx", "fma4"}, "FMA4") {};

			void compilePayload(std::map<std::string, unsigned> proportion) override;
			std::list<std::string> getAvailableInstructions(void) override;
			void init(...) override;
			void highLoadFunction(...) override;
	};
}

#endif
