#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_X86ENVIRONMENT_HPP
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_X86ENVIRONMENT_HPP

#include <firestarter/Environment/Environment.hpp>

#include <firestarter/Environment/X86/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Platform/KnightsLandingConfig.hpp>
#include <firestarter/Environment/X86/Platform/SkylakeConfig.hpp>
#include <firestarter/Environment/X86/Platform/SkylakeSPConfig.hpp>
#include <firestarter/Environment/X86/Platform/HaswellConfig.hpp>
#include <firestarter/Environment/X86/Platform/HaswellEPConfig.hpp>
#include <firestarter/Environment/X86/Platform/SandyBridgeConfig.hpp>
#include <firestarter/Environment/X86/Platform/SandyBridgeEPConfig.hpp>
#include <firestarter/Environment/X86/Platform/NehalemConfig.hpp>
#include <firestarter/Environment/X86/Platform/NehalemEPConfig.hpp>
#include <firestarter/Environment/X86/Platform/BulldozerConfig.hpp>

#include <functional>

extern "C" {
#include <firestarter/Compat/util.h>
}

#define REGISTER_PLATFORMCONFIG(NAME) \
	[](llvm::StringMap<bool> *supportedFeatures, unsigned family, unsigned model, unsigned threads) -> \
		platform::PlatformConfig * { return new platform::NAME(supportedFeatures, family, model, threads); }

namespace firestarter::environment::x86 {

	class X86Environment : public Environment {
		public:
			X86Environment() : Environment() {};
			~X86Environment() {};

			void evaluateFunctions(void) override;
			int selectFunction(unsigned functionId) override;
			void printFunctionSummary(void) override;

		private:
			int getCpuClockrate(void) override;

			const std::list<std::function<platform::PlatformConfig *(llvm::StringMap<bool> *, unsigned, unsigned, unsigned)>> platformConfigsCtor = {
				REGISTER_PLATFORMCONFIG(KnightsLandingConfig),
				REGISTER_PLATFORMCONFIG(SkylakeConfig),
				REGISTER_PLATFORMCONFIG(SkylakeSPConfig),
				REGISTER_PLATFORMCONFIG(HaswellConfig),
				REGISTER_PLATFORMCONFIG(HaswellEPConfig),
				REGISTER_PLATFORMCONFIG(SandyBridgeConfig),
				REGISTER_PLATFORMCONFIG(SandyBridgeEPConfig),
				REGISTER_PLATFORMCONFIG(NehalemConfig),
				REGISTER_PLATFORMCONFIG(NehalemEPConfig),
				REGISTER_PLATFORMCONFIG(BulldozerConfig),
			};

			std::list<platform::PlatformConfig *> platformConfigs;
	};

}

#endif
