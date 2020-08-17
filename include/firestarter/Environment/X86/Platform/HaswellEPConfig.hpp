#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_HASWELLEPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_HASWELLEPCONFIG_H

#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class HaswellEPConfig : public X86PlatformConfig {

public:
  HaswellEPConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
                  unsigned model, unsigned threads)
      : X86PlatformConfig("HSW_XEONEP", 6, {63, 79}, {1, 2},
                          {32768, 262144, 2621440}, 104857600, family, model,
                          threads,
                          new payload::FMAPayload(supportedFeatures)){};
  ~HaswellEPConfig(){};

  std::vector<std::pair<std::string, unsigned>> getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>(
        {{"RAM_L", 2}, {"L3_LS", 1}, {"L2_LS", 9}, {"L1_LS", 79}, {"REG", 35}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
