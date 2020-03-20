#ifndef INCLUDE_FIRESTARTER_UTIL_H
#define INCLUDE_FIRESTARTER_UTIL_H

#if ((defined (__x86_64__))||(defined (__x86_64))||(defined (x86_64))||(defined (__i386__))||(defined (__i386))||(defined (i386))||(defined (__i486__))||(defined (__i486))||(defined (i486))||(defined (__i586__))||(defined (__i586))||(defined (i586))||(defined (__i686__))||(defined (__i686))||(defined (i686)))
	#define __ARCH_X86
#else
	#define __ARCH_UNKNOWN
#endif

#define _HW_DETECT_MAX_OUTPUT 512

#endif
