#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_BULLDOZERCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_BULLDOZERCONFIG_H

#include <firestarter/Environment/X86/Payload/FMA4Payload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class BulldozerConfig : public X86PlatformConfig {

public:
  BulldozerConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
                  unsigned model, unsigned threads)
      : X86PlatformConfig("BLD_OPTERON", 21, {1, 2, 3}, {1},
                          {16384, 1048576, 786432}, 104857600, family, model,
                          threads,
                          new payload::FMA4Payload(supportedFeatures)){};
  ~BulldozerConfig(){};

  std::map<std::string, unsigned> getDefaultPayloadSettings(void) override {
    return std::map<std::string, unsigned>(
        {{"RAM_L", 1}, {"L3_L", 1}, {"L2_LS", 5}, {"L1_L", 90}, {"REG", 45}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
