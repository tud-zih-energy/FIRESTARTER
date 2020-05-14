#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_PLATFORMCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_PLATFORMCONFIG_H

#include <llvm/ADT/StringMap.h>

#include <firestarter/Environment/X86/Payload/Payload.hpp>

#include <initializer_list>
#include <map>
#include <algorithm>
#include <string>

namespace firestarter::environment::x86::platform {

	class PlatformConfig {
		private:
			std::string _name;
			unsigned _family;
			std::list<unsigned> _models;
			std::list<unsigned> _threads;
			unsigned _currentFamily;
			unsigned _currentModel;
			unsigned _currentThreads;
			payload::Payload *_payload;

		public:
			PlatformConfig(std::string name, unsigned family, std::initializer_list<unsigned> models, std::initializer_list<unsigned> threads, unsigned currentFamily, unsigned currentModel, unsigned currentThreads, payload::Payload *payload) :
				_name(name), _family(family), _models(models), _threads(threads), _currentFamily(currentFamily), _currentModel(currentModel), _currentThreads(currentThreads), _payload(payload) {};

			~PlatformConfig() {
				delete _payload;
			}

			payload::Payload *getPayload(void) {
				return _payload;
			}

			std::map<unsigned, std::string> getThreadMap(void) {
				std::map<unsigned, std::string> threadMap;

				for (auto const& thread : _threads) {
					std::stringstream functionName;
					functionName << "FUNC_" << _name << "_" << _payload->getName() << "_" << thread << "T";
					threadMap[thread] = functionName.str();
				}

				return threadMap;
			}

			std::string getName(void) {
				return _name;
			}

			bool isAvailable(void) {
				return _payload->isAvailable();
			}

			bool isDefault(void) {
				return _family == _currentFamily &&
					(std::find(_models.begin(), _models.end(), _currentModel) != _models.end()) &&
					(std::find(_threads.begin(), _threads.end(), _currentModel) != _threads.end()) &&
					isAvailable();
			}
	};

}

#endif
