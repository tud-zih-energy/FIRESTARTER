#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_AVXPAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_AVXPAYLOAD_H

#include <firestarter/Environment/X86/Payload/Payload.hpp>

namespace firestarter::environment::x86::payload {
	class AVXPayload : public Payload {

		private:
	
		public:
			AVXPayload(llvm::StringMap<bool> *supportedFeatures) : Payload(supportedFeatures, {"avx"}) {};

			std::string getName(void) override {
				return "AVX";
			}

			void compilePayload(llvm::StringMap<unsigned> proportion);
			std::list<std::string> getAvailableInstructions(void);
			void init(...);
			void highLoadFunction(...);
	};
}

#endif
