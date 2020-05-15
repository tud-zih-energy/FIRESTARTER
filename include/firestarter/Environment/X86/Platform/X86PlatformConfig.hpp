#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_X86PLATFORMCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_X86PLATFORMCONFIG_H

#include <llvm/ADT/StringMap.h>

#include <firestarter/Environment/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

#include <initializer_list>
#include <map>
#include <algorithm>
#include <string>

namespace firestarter::environment::x86::platform {

    class X86PlatformConfig : public environment::platform::PlatformConfig {
		private:
			unsigned _family;
			std::list<unsigned> _models;
			std::list<unsigned> _threads;
			unsigned _currentFamily;
			unsigned _currentModel;
			unsigned _currentThreads;

		public:
			X86PlatformConfig(std::string name, unsigned family, std::initializer_list<unsigned> models, std::initializer_list<unsigned> threads, unsigned currentFamily, unsigned currentModel, unsigned currentThreads, payload::X86Payload *payload) :
			    PlatformConfig(name, threads, payload), _family(family), _models(models), _currentFamily(currentFamily), _currentModel(currentModel), _currentThreads(currentThreads) {};

			~X86PlatformConfig() {
				delete payload;
			}

			bool isDefault(unsigned thread) override {
				return _family == _currentFamily &&
					(std::find(_models.begin(), _models.end(), _currentModel) != _models.end()) &&
					(std::find(_threads.begin(), _threads.end(), _currentThreads) != _threads.end()) &&
					_currentThreads == thread &&
					isAvailable();
			}
	};

}

#endif
