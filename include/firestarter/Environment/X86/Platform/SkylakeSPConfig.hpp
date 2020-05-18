#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKESPCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PLATFORM_SKYLAKESPCONFIG_H

#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>
#include <firestarter/Environment/X86/Platform/X86PlatformConfig.hpp>

namespace firestarter::environment::x86::platform {
class SkylakeSPConfig : public X86PlatformConfig {

public:
  SkylakeSPConfig(llvm::StringMap<bool> *supportedFeatures, unsigned family,
                  unsigned model, unsigned threads)
      : X86PlatformConfig("SKL_XEONEP", 6, {85}, {1, 2},
                          {32768, 1048576, 1441792}, 1048576000, family, model,
                          threads,
                          new payload::AVX512Payload(supportedFeatures)){};

  ~SkylakeSPConfig(){};

  std::map<std::string, unsigned> getDefaultPayloadSettings(void) override {
    return std::map<std::string, unsigned>({{"RAM_S", 3},
                                            {"RAM_P", 1},
                                            {"L3_S", 1},
                                            {"L3_P", 1},
                                            {"L2_S", 4},
                                            {"L2_L", 70},
                                            {"L1_S", 0},
                                            {"L1_L", 40},
                                            {"L1_BROADCAST", 120},
                                            {"REG", 160}});
  }
};
} // namespace firestarter::environment::x86::platform

#endif
