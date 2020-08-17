#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_AVX512PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_AVX512PAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class AVX512Payload : public X86Payload {
public:
  AVX512Payload(llvm::StringMap<bool> *supportedFeatures)
      : X86Payload(supportedFeatures, {"avx512f"}, "AVX512"){};

  int compilePayload(std::vector<std::pair<std::string, unsigned>> proportion,
                     std::list<unsigned> dataCacheBufferSize,
                     unsigned ramBufferSize, unsigned thread,
                     unsigned numberOfLines) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;

  firestarter::environment::payload::Payload *clone(void) override {
    return new AVX512Payload(this->supportedFeatures);
  };

private:
  const std::map<std::string, unsigned> instructionFlops = {
      {"REG", 32},   {"L1_L", 32},  {"L1_BROADCAST", 16}, {"L1_S", 16},
      {"L1_LS", 16}, {"L2_L", 32},  {"L2_S", 16},         {"L2_LS", 16},
      {"L3_L", 32},  {"L3_S", 16},  {"L3_LS", 16},        {"L3_P", 16},
      {"RAM_L", 32}, {"RAM_S", 16}, {"RAM_LS", 16},       {"RAM_P", 16}};

  const std::map<std::string, unsigned> instructionMemory = {
      {"RAM_L", 64}, {"RAM_S", 128}, {"RAM_LS", 128}, {"RAM_P", 64}};
};
} // namespace firestarter::environment::x86::payload

#endif
