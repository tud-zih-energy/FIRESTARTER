#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>
#include <firestarter/Logging/Log.hpp>

#include <utility>

using namespace firestarter::environment::x86::payload;
using namespace asmjit;

int FMAPayload::compilePayload(std::map<std::string, unsigned> proportion) {
  CodeHolder code;
  code.init(this->rt.codeInfo());

  asmjit::x86::Assembler a(&code);
  a.mov(asmjit::x86::eax, 1);
  a.ret();

  Error err = this->rt.add(&this->loadFunction, &code);
  if (err) {
    log::error() << "Error: Asmjit adding Assembler to JitRuntime failed in "
                 << __FILE__ << " at " << __LINE__;
    return EXIT_FAILURE;
  }

  // test delete the function pointer from the runtime.
  // this has to be done before the function call to the current one
  this->rt.release(&this->loadFunction);

  // create a std::function pointer and bind func to it

  return EXIT_SUCCESS;
}

std::list<std::string> FMAPayload::getAvailableInstructions(void) {}

void FMAPayload::init(unsigned long long *memoryAddr,
                      unsigned long long bufferSize) {
  X86Payload::init(memoryAddr, bufferSize, 0.27948995982e-4, 0.27948995982e-4);
}

void FMAPayload::highLoadFunction(...) {
  log::debug() << "The result is: " << this->loadFunction();
}
