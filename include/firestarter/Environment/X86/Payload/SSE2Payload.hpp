#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_SSE2PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_SSE2PAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class SSE2Payload : public X86Payload {
public:
  SSE2Payload(llvm::StringMap<bool> *supportedFeatures)
      : X86Payload(supportedFeatures, {"sse2"}, "SSE2"){};

  int compilePayload(std::map<std::string, unsigned> proportion,
                     std::list<unsigned> dataCacheBufferSize,
                     unsigned ramBufferSize, unsigned thread,
                     unsigned numberOfLines) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;
  unsigned long long highLoadFunction(unsigned long long *addrMem,
                                      volatile unsigned long long *addrHigh,
                                      unsigned long long iterations) override;

  firestarter::environment::payload::Payload *clone(void) override {
    return new SSE2Payload(this->supportedFeatures);
  };
};
} // namespace firestarter::environment::x86::payload

#endif
