#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_KNIGHTSLANDINGCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_KNIGHTSLANDINGCONFIG_H

#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class KnightsLandingConfig : public X86PlatformConfig {

public:
  KnightsLandingConfig(llvm::StringMap<bool> *supportedFeatures,
                       unsigned family, unsigned model, unsigned threads)
      : X86PlatformConfig("KNL_XEONPHI", 6, {87}, {4},
                          {32768, 524288, 236279125}, 26214400, family, model,
                          threads,
                          new payload::AVX512Payload(supportedFeatures)){};
  ~KnightsLandingConfig(){};

  std::vector<std::pair<std::string, unsigned>> getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>(
        {{"RAM_P", 3}, {"L2_S", 8}, {"L1_L", 40}, {"REG", 10}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
