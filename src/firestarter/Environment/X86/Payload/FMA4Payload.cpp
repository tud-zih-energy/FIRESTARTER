#include <firestarter/Environment/X86/Payload/FMA4Payload.hpp>

using namespace firestarter::environment::x86::payload;

int FMA4Payload::compilePayload(std::map<std::string, unsigned> proportion) {}

std::list<std::string> FMA4Payload::getAvailableInstructions(void) {}

void FMA4Payload::init(unsigned long long *memoryAddr,
                       unsigned long long bufferSize) {
  X86Payload::init(memoryAddr, bufferSize, 0.27948995982e-4, 0.27948995982e-4);
}

void FMA4Payload::highLoadFunction(...) {}
