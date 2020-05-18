#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGEEPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SANDYBRIDGEEPCONFIG_H

#include <firestarter/Environment/X86/Payload/AVXPayload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class SandyBridgeEPConfig : public X86PlatformConfig {

public:
  SandyBridgeEPConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
                      unsigned model, unsigned threads)
      : X86PlatformConfig("SNB_XEONEP", 6, {45, 62}, {1, 2},
                          {32768, 262144, 2621440}, 104857600, family, model,
                          threads,
                          new payload::AVXPayload(supportedFeatures)){};
  ~SandyBridgeEPConfig(){};

  std::map<std::string, unsigned> getDefaultPayloadSettings(void) override {
    return std::map<std::string, unsigned>({{"RAM_L", 3},
                                            {"L3_LS", 2},
                                            {"L2_LS", 10},
                                            {"L1_LS", 90},
                                            {"REG", 30}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
