#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class FMAPayload : public X86Payload {
public:
  FMAPayload(llvm::StringMap<bool> *supportedFeatures)
      : X86Payload(supportedFeatures, {"avx", "fma"}, "FMA"){};

  void compilePayload(std::map<std::string, unsigned> proportion) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(...) override;
  void highLoadFunction(...) override;
};
} // namespace firestarter::environment::x86::payload

#endif
