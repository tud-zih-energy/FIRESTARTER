#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>

using namespace firestarter::environment::x86::payload;

int AVX512Payload::compilePayload(std::map<std::string, unsigned> proportion,
                                  std::list<unsigned> dataCacheBufferSize,
                                  unsigned ramBufferSize, unsigned thread,
                                  unsigned numberOfLines) {}

std::list<std::string> AVX512Payload::getAvailableInstructions(void) {}

void AVX512Payload::init(unsigned long long *memoryAddr,
                         unsigned long long bufferSize) {
  X86Payload::init(memoryAddr, bufferSize, 0.27948995982e-4, 0.27948995982e-4);
}

unsigned long long
AVX512Payload::highLoadFunction(unsigned long long *addrMem,
                                volatile unsigned long long *addrHigh,
                                unsigned long long iterations){};
