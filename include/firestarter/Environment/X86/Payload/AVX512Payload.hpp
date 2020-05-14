#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_AVX512PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_AVX512PAYLOAD_H

#include <firestarter/Environment/X86/Payload/Payload.hpp>

namespace firestarter::environment::x86::payload {
	class AVX512Payload : public Payload {

		private:
	
		public:
			AVX512Payload(llvm::StringMap<bool> *supportedFeatures) : Payload(supportedFeatures, {"avx512f"}) {};

			std::string getName(void) override {
				return "AVX512";
			}

			void compilePayload(llvm::StringMap<unsigned> proportion);
			std::list<std::string> getAvailableInstructions(void);
			void init(...);
			void highLoadFunction(...);
	};
}

#endif
