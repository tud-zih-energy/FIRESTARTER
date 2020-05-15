#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_PLATFORM_PLATFORMCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_PLATFORM_PLATFORMCONFIG_H

#include <llvm/ADT/StringMap.h>

#include <firestarter/Environment/Payload/Payload.hpp>

#include <initializer_list>
#include <map>
#include <algorithm>
#include <sstream>
#include <string>

namespace firestarter::environment::platform {

	class PlatformConfig {
		private:
			std::string _name;
			std::list<unsigned> _threads;

		public:
			PlatformConfig(std::string name, std::list<unsigned> threads, payload::Payload *payload) :
				_name(name), _threads(threads), payload(payload) {};
			~PlatformConfig() {};

			payload::Payload *payload;

			std::map<unsigned, std::string> getThreadMap(void) {
				std::map<unsigned, std::string> threadMap;

				for (auto const& thread : _threads) {
					std::stringstream functionName;
					functionName << "FUNC_" << _name << "_" << payload->getName() << "_" << thread << "T";
					threadMap[thread] = functionName.str();
				}

				return threadMap;
			}

			std::string getName(void) {
				return _name;
			}

			bool isAvailable(void) {
				return payload->isAvailable();
			}

			virtual bool isDefault(unsigned thread) =0;
	};

}

#endif
