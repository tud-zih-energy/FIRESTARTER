#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_PAYLOAD_FMAPAYLOAD_H

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {
class FMAPayload : public X86Payload {
public:
  FMAPayload(llvm::StringMap<bool> *supportedFeatures)
      : X86Payload(supportedFeatures, {"avx", "fma"}, "FMA"){};

  int compilePayload(std::vector<std::pair<std::string, unsigned>> proportion,
                     std::list<unsigned> dataCacheBufferSize,
                     unsigned ramBufferSize, unsigned thread,
                     unsigned numberOfLines) override;
  std::list<std::string> getAvailableInstructions(void) override;
  void init(unsigned long long *memoryAddr,
            unsigned long long bufferSize) override;

  firestarter::environment::payload::Payload *clone(void) override {
    return new FMAPayload(this->supportedFeatures);
  };

private:
  const std::map<std::string, unsigned> instructionFlops = {
      {"REG", 16},  {"L1_L", 16},     {"L1_2L", 16},      {"L1_S", 8},
      {"L1_LS", 8}, {"L1_LS_256", 8}, {"L1_2LS_256", 16}, {"L2_L", 16},
      {"L2_S", 8},  {"L2_LS", 8},     {"L2_LS_256", 8},   {"L2_2LS_256", 16},
      {"L3_L", 16}, {"L3_S", 8},      {"L3_LS", 8},       {"L3_LS_256", 8},
      {"L3_P", 8},  {"RAM_L", 16},    {"RAM_S", 8},       {"RAM_LS", 8},
      {"RAM_P", 8}};

  const std::map<std::string, unsigned> instructionMemory = {
      {"RAM_L", 64}, {"RAM_S", 128}, {"RAM_LS", 128}, {"RAM_P", 64}};

};
} // namespace firestarter::environment::x86::payload

#endif
