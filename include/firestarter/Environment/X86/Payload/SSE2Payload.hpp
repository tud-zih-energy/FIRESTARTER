#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_SSE2PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_SSE2PAYLOAD_H

#include <firestarter/Environment/X86/Payload/Payload.hpp>

namespace firestarter::environment::x86::payload {
	class SSE2Payload : public Payload {

		private:
	
		public:
			SSE2Payload(llvm::StringMap<bool> *supportedFeatures) : Payload(supportedFeatures, {"sse2"}) {};

			std::string getName(void) override {
				return "SSE2";
			}

			void compilePayload(llvm::StringMap<unsigned> proportion);
			std::list<std::string> getAvailableInstructions(void);
			void init(...);
			void highLoadFunction(...);
	};
}

#endif
