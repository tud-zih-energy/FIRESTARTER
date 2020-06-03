#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class FMAPayload : public X86Payload {
public:
  FMAPayload(llvm::StringMap<bool> *supportedFeatures)
      : X86Payload(supportedFeatures, {"avx", "fma"}, "FMA"){};

  int compilePayload(std::map<std::string, unsigned> proportion,
                     std::list<unsigned> dataCacheBufferSize,
                     unsigned ramBufferSize, unsigned thread,
                     unsigned numberOfLines) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;

  firestarter::environment::payload::Payload *clone(void) override {
    return new FMAPayload(this->supportedFeatures);
  };
};
} // namespace firestarter::environment::x86::payload

#endif
