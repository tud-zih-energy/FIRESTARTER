#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_AVX512PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_AVX512PAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class AVX512Payload : public X86Payload {
public:
  AVX512Payload(llvm::StringMap<bool> *supportedFeatures)
      : X86Payload(supportedFeatures, {"avx512f"}, "AVX512"){};

  int compilePayload(std::map<std::string, unsigned> proportion,
                     std::list<unsigned> dataCacheBufferSize,
                     unsigned ramBufferSize, unsigned thread,
                     unsigned numberOfLines) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;

  firestarter::environment::payload::Payload *clone(void) override {
    return new AVX512Payload(this->supportedFeatures);
  };
};
} // namespace firestarter::environment::x86::payload

#endif
