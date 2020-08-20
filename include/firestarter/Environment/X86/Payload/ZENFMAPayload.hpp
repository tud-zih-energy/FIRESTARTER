#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_ZENFMAPAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_ZENFMAPAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class ZENFMAPayload : public X86Payload {
public:
  ZENFMAPayload(llvm::StringMap<bool> *supportedFeatures)
      : X86Payload(supportedFeatures, {"avx", "fma"}, "ZENFMA"){};

  int compilePayload(std::vector<std::pair<std::string, unsigned>> proportion,
                     std::list<unsigned> dataCacheBufferSize,
                     unsigned ramBufferSize, unsigned thread,
                     unsigned numberOfLines) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;

  firestarter::environment::payload::Payload *clone(void) override {
    return new ZENFMAPayload(this->supportedFeatures);
  };

private:
  const std::map<std::string, unsigned> instructionFlops = {
      {"REG", 8}, {"L1_LS", 8}, {"L2_L", 8}, {"L3_L", 8}, {"RAM_L", 8}};

  const std::map<std::string, unsigned> instructionMemory = {{"RAM_L", 64}};
};
} // namespace firestarter::environment::x86::payload

#endif
