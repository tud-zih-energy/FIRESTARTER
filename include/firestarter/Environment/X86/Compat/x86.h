#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_X86_COMPAT_X86_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_X86_COMPAT_X86_H

#include <firestarter/Environment/X86/Compat/util.h>

#if ((defined (__x86_64__))||(defined (__x86_64))||(defined (x86_64)))
	#define _64_BIT
#elif ((defined (__i386__))||(defined (__i386))||(defined (i386))||(defined (__i486__))||(defined (__i486))||(defined (i486))||(defined (__i586__))||(defined (__i586))||(defined (i586))||(defined (__i686__))||(defined (__i686))||(defined (i686)))
	#define _32_BIT
#endif

static void x86_cpuid(unsigned long long *a, unsigned long long *b, unsigned long long *c, unsigned long long *d);
static int x86_has_cpuid();

unsigned long long x86_timestamp(void);
int x86_has_rdtsc(void);
int x86_has_invariant_rdtsc(const char *vendor);

#endif
