#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMA4PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMA4PAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {

class FMA4Payload : public X86Payload {
public:
  FMA4Payload(llvm::StringMap<bool> *supportedFeatures)
      : X86Payload(supportedFeatures, {"avx", "fma4"}, "FMA4"){};

  int compilePayload(std::vector<std::pair<std::string, unsigned>> proportion,
                     std::list<unsigned> dataCacheBufferSize,
                     unsigned ramBufferSize, unsigned thread,
                     unsigned numberOfLines) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;

  firestarter::environment::payload::Payload *clone(void) override {
    return new FMA4Payload(this->supportedFeatures);
  };

private:
  const std::map<std::string, unsigned> instructionFlops = {
      {"REG", 8},  {"L1_L", 12}, {"L1_S", 8},  {"L1_LS", 8},  {"L2_L", 8},
      {"L2_S", 4}, {"L2_LS", 4}, {"L3_L", 8},  {"L3_S", 4},   {"L3_LS", 4},
      {"L3_P", 4}, {"RAM_L", 8}, {"RAM_S", 4}, {"RAM_LS", 4}, {"RAM_P", 4}};

  const std::map<std::string, unsigned> instructionMemory = {
      {"RAM_L", 64}, {"RAM_S", 128}, {"RAM_LS", 128}, {"RAM_P", 64}};
};
} // namespace firestarter::environment::x86::payload

#endif
