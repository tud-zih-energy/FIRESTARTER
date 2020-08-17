#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKECONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKECONFIG_H

#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class SkylakeConfig : public X86PlatformConfig {

public:
  SkylakeConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
                unsigned model, unsigned threads)
      : X86PlatformConfig("SKL_COREI", 6, {78, 94}, {1, 2},
                          {32768, 262144, 1572864}, 104857600, family, model,
                          threads,
                          new payload::FMAPayload(supportedFeatures)){};
  ~SkylakeConfig(){};

	std::vector<std::pair<std::string, unsigned>> getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_L", 3},
                                            {"L3_LS_256", 5},
                                            {"L2_LS_256", 18},
                                            {"L1_2LS_256", 78},
                                            {"REG", 40}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
