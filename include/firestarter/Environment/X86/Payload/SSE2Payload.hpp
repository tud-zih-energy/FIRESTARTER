#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_SSE2PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_SSE2PAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
	class SSE2Payload : public X86Payload {
		public:
			SSE2Payload(llvm::StringMap<bool> *supportedFeatures) : X86Payload(supportedFeatures, {"sse2"}, "SSE2") {};

			void compilePayload(std::map<std::string, unsigned> proportion) override;
			std::list<std::string> getAvailableInstructions(void) override;
			void init(...) override;
			void highLoadFunction(...) override;
	};
}

#endif
