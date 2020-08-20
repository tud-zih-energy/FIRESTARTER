#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_NAPLESCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_NAPLESCONFIG_H

#include <firestarter/Environment/X86/Payload/ZENFMAPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class NaplesConfig : public X86PlatformConfig {

public:
  NaplesConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
               unsigned model, unsigned threads)
      : X86PlatformConfig("ZEN_EPYC", 23, {1, 8, 17, 24}, {1, 2},
                          {65536, 524288, 2097152}, 104857600, family, model,
                          threads,
                          new payload::ZENFMAPayload(supportedFeatures)){};
  ~NaplesConfig(){};

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_L", 8},
                                                          {"L3_L", 33},
                                                          {"L2_L", 81},
                                                          {"L1_LS", 79},
                                                          {"REG", 100}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
