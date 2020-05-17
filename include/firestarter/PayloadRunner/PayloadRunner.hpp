#ifndef INCLUDE_FIRESTARTER_PAYLOADRUNNER_PAYLOADRUNNER_HPP
#define INCLUDE_FIRESTARTER_PAYLOADRUNNER_PAYLOADRUNNER_HPP

#include <firestarter/Environment/Platform/PlatformConfig.hpp>

namespace firestarter::payloadrunner {

	class PayloadRunner {
		public:
			PayloadRunner(environment::platform::PlatformConfig *config, unsigned threads) : config(config), threads(threads) {};
			~PayloadRunner() {};

			int init(void) {
				return EXIT_SUCCESS;
			}

		private:
			environment::platform::PlatformConfig *config;
			unsigned threads;
	};

}

#endif
