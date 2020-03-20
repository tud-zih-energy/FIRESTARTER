#ifndef INCLUDE_FIRESTARTER_X86_H
#define INCLUDE_FIRESTARTER_X86_H

#include <firestarter/util.h>

#if defined (__ARCH_X86)
	#if ((defined (__x86_64__))||(defined (__x86_64))||(defined (x86_64)))
		#define _64_BIT
	#elif ((defined (__i386__))||(defined (__i386))||(defined (i386))||(defined (__i486__))||(defined (__i486))||(defined (i486))||(defined (__i586__))||(defined (__i586))||(defined (i586))||(defined (__i686__))||(defined (__i686))||(defined (i686)))
		#define _32_BIT
	#endif
#endif

void cpuid(unsigned long long *a, unsigned long long *b, unsigned long long *c, unsigned long long *d);
int has_cpuid(void);
unsigned long long timestamp(void);
int has_rdtsc(void);

#endif
