#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_PAYLOAD_PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_PAYLOAD_PAYLOAD_H

#include <llvm/ADT/StringMap.h>

#include <initializer_list>
#include <list>
#include <string>

namespace firestarter::environment::payload {

	class Payload {
		private:
			std::string _name;

		public:
			Payload(std::string name) : _name(name) {};
			~Payload() {};

			std::string getName(void) {
				return _name;
			}

			virtual bool isAvailable(void) =0;

			virtual void lowLoadFunction(...) =0;

			virtual void compilePayload(llvm::StringMap<unsigned> proportion) =0;
			virtual std::list<std::string> getAvailableInstructions(void) =0;
			virtual void init(...) =0;
			virtual void highLoadFunction(...) =0;
	};

}

#endif
