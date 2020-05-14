#include <firestarter/Environment/X86/Payload/Payload.hpp>

using namespace firestarter::environment::x86::payload;

void Payload::lowLoadFunction(...) {
#if 0
	int nap;

	nap = period / 100;
	__asm__ __volatile__ ("mfence;"
			"cpuid;" ::: "eax", "ebx", "ecx", "edx");
	// while signal low load
	while(... LOAD_LOW){
		__asm__ __volatile__ ("mfence;"
				"cpuid;" ::: "eax", "ebx", "ecx", "edx");
		usleep(nap);
		__asm__ __volatile__ ("mfence;"
				"cpuid;" ::: "eax", "ebx", "ecx", "edx");
	}
#endif
}
