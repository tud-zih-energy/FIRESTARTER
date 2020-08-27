#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_ROMECONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_ROMECONFIG_H

#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class RomeConfig : public X86PlatformConfig {

public:
  RomeConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
               unsigned model, unsigned threads)
      : X86PlatformConfig("ZEN_2_EPYC", 23, {49}, {1,2},
                          {32768, 524288, 2097152}, 104857600, family, model,
                          threads,
                          new payload::FMAPayload(supportedFeatures)){};
  ~RomeConfig(){};

  std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_L", 8},
                                                          {"L3_L", 53},
                                                          {"L2_L", 37},
                                                          {"L1_2LS_256", 33},
                                                          {"L1_LS_256", 66},
                                                          {"REG", 22}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
