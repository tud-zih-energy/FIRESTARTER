#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H

#include <firestarter/Environment/X86/Payload/Payload.hpp>

namespace firestarter::environment::x86::payload {
	class FMAPayload : public Payload {

		private:
	
		public:
			FMAPayload(llvm::StringMap<bool> *supportedFeatures) : Payload(supportedFeatures, {"avx", "fma"}) {};

			std::string getName(void) override {
				return "FMA";
			}

			void compilePayload(llvm::StringMap<unsigned> proportion);
			std::list<std::string> getAvailableInstructions(void);
			void init(...);
			void highLoadFunction(...);
	};
}

#endif
