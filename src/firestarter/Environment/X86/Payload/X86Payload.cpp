#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

using namespace firestarter::environment::x86::payload;

void X86Payload::lowLoadFunction(...) {
  int nap;

  //	nap = period / 100;
  __asm__ __volatile__("mfence;"
                       "cpuid;" ::
                           : "eax", "ebx", "ecx", "edx");
  // while signal low load
  while (true) {
    __asm__ __volatile__("mfence;"
                         "cpuid;" ::
                             : "eax", "ebx", "ecx", "edx");
    //		usleep(nap);
    __asm__ __volatile__("mfence;"
                         "cpuid;" ::
                             : "eax", "ebx", "ecx", "edx");
  }
}

void X86Payload::init(unsigned long long *memoryAddr,
                      unsigned long long bufferSize, double firstValue,
                      double lastValue) {
  int i;

  for (i = 0; i < INIT_BLOCKSIZE; i++)
    *((double *)(memoryAddr + i)) = 0.25 + (double)i * 8.0 * firstValue;
  for (; i <= bufferSize - INIT_BLOCKSIZE; i += INIT_BLOCKSIZE)
    std::memcpy(memoryAddr + i, memoryAddr + i - INIT_BLOCKSIZE,
                sizeof(unsigned long long) * INIT_BLOCKSIZE);
  for (; i < bufferSize; i++)
    *((double *)(memoryAddr + i)) = 0.25 + (double)i * 8.0 * lastValue;
}

unsigned long long
X86Payload::highLoadFunction(unsigned long long *addrMem,
                             volatile unsigned long long *addrHigh,
                             unsigned long long iterations) {
  return this->loadFunction(addrMem, addrHigh, iterations);
}
