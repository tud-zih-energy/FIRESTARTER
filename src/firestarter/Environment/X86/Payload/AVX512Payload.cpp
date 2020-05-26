#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>

using namespace firestarter::environment::x86::payload;

int AVX512Payload::compilePayload(std::map<std::string, unsigned> proportion) {}

std::list<std::string> AVX512Payload::getAvailableInstructions(void) {}

void AVX512Payload::init(unsigned long long *memoryAddr,
                         unsigned long long bufferSize) {
  X86Payload::init(memoryAddr, bufferSize, 0.27948995982e-4, 0.27948995982e-4);
}

void AVX512Payload::highLoadFunction(...) {}
