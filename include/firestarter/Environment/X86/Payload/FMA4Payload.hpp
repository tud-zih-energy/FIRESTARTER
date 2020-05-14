#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMA4PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMA4PAYLOAD_H

#include <firestarter/Environment/X86/Payload/Payload.hpp>

namespace firestarter::environment::x86::payload {

	class FMA4Payload : public Payload {
		private:
	
		public:
			FMA4Payload(llvm::StringMap<bool> *supportedFeatures) : Payload(supportedFeatures, {"avx", "fma4"}) {};

			std::string getName(void) override {
				return "FMA4";
			}

			void compilePayload(llvm::StringMap<unsigned> proportion);
			std::list<std::string> getAvailableInstructions(void);
			void init(...);
			void highLoadFunction(...);
	};
}

#endif
