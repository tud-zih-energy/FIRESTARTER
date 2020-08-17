#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_HASWELLCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_HASWELLCONFIG_H

#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class HaswellConfig : public X86PlatformConfig {

public:
  HaswellConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
                unsigned model, unsigned threads)
      : X86PlatformConfig("HSW_COREI", 6, {60, 61, 69, 70, 71}, {1, 2},
                          {32768, 262144, 1572864}, 104857600, family, model,
                          threads,
                          new payload::FMAPayload(supportedFeatures)){};
  ~HaswellConfig(){};

  std::vector<std::pair<std::string, unsigned>> getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>(
        {{"RAM_L", 2}, {"L3_LS", 3}, {"L2_LS", 9}, {"L1_LS", 90}, {"REG", 40}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
