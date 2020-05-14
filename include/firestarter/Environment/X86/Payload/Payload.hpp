#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_PAYLOAD_H

#include <llvm/ADT/StringMap.h>

#include <firestarter/Logging/Log.hpp>

#include <initializer_list>
#include <list>
#include <string>

namespace firestarter::environment::x86::payload {

	class Payload {
		private:
			// we can use this to check, if our platform support this payload
			llvm::StringMap<bool> *_supportedFeatures;
			std::list<std::string> _featureRequests;
	
		public:
			Payload(llvm::StringMap<bool> *supportedFeatures, std::initializer_list<std::string> featureRequests) :
				_supportedFeatures(supportedFeatures), _featureRequests(featureRequests) {};
			~Payload() {};

			virtual std::string getName(void) =0;

			bool isAvailable(void) {
				bool available = true;

				for (std::string feature : _featureRequests) {
					available &= (*_supportedFeatures)[feature];
				}

				return available;
			};

			// A generic implemenation for all x86 payloads
			// use cpuid and usleep as low load
			void lowLoadFunction(...);

			// specific implementation (FMA, FMA4, AVX512 etc.)
			virtual void compilePayload(llvm::StringMap<unsigned> proportion) =0;
			virtual std::list<std::string> getAvailableInstructions(void) =0;
			virtual void init(...) =0;
			virtual void highLoadFunction(...) =0;
	};

}

#endif
