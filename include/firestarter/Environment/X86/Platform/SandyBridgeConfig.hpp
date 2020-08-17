#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGECONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGECONFIG_H

#include <firestarter/Environment/X86/Payload/AVXPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class SandyBridgeConfig : public X86PlatformConfig {

public:
  SandyBridgeConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
                    unsigned model, unsigned threads)
      : X86PlatformConfig("SNB_COREI", 6, {42, 58}, {1, 2},
                          {32768, 262144, 1572864}, 104857600, family, model,
                          threads,
                          new payload::AVXPayload(supportedFeatures)){};
  ~SandyBridgeConfig(){};

  std::vector<std::pair<std::string, unsigned>> getDefaultPayloadSettings(void) override {
    return std::vector<std::pair<std::string, unsigned>>({{"RAM_L", 2},
                                            {"L3_LS", 4},
                                            {"L2_LS", 10},
                                            {"L1_LS", 90},
                                            {"REG", 45}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
