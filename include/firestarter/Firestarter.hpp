#ifndef INCLUDE_FIRESTARTER_FIRESTARTER_HPP
#define INCLUDE_FIRESTARTER_FIRESTARTER_HPP

#include <firestarter/ThreadData.hpp>

#include <firestarter/Environment/X86/X86Environment.hpp>
#include <firestarter/Environment/X86/Platform/PlatformConfig.hpp>

#include <list>
#include <string>

extern "C" {
#include <pthread.h>
}

namespace firestarter {
	
	class Firestarter {
		public:
			Firestarter(void) {
#if defined(__i386__) || defined(_M_IX86) || \
		defined(__x86_64__) || defined(_M_X64)
				environment = new environment::x86::X86Environment();
#else
#error "FIRESTARTER is not implemented for this ISA"
#endif
			};

			~Firestarter(void) {
				delete environment;
			};

			environment::Environment *environment;

			void init(void);

		private:
			// ThreadWorker.cpp
			static void *threadWorker(void *threadData);

			// ThreadWorker.cpp
			pthread_t *threads;
			std::list<ThreadData *> threadData;
	};

}

#endif
