#include <firestarter/Environment/X86/Compat/x86.h>

#include <string.h>

#if defined _64_BIT

static void x86_cpuid(unsigned long long *a, unsigned long long *b, unsigned long long *c, unsigned long long *d)
{
    unsigned long long reg_a,reg_b,reg_c,reg_d;
 
    __asm__ __volatile__(
        "cpuid;"
        : "=a" (reg_a), "=b" (reg_b), "=c" (reg_c), "=d" (reg_d)
        : "a" (*a), "b" (*b), "c" (*c), "d" (*d)
    );
    *a=reg_a;
    *b=reg_b;
    *c=reg_c;
    *d=reg_d;
}

static int x86_has_cpuid()
{
    // all 64 Bit x86 CPUs support CPUID
    return 1;
}

unsigned long long x86_timestamp()
{
    unsigned long long reg_a,reg_d;

    if (!x86_has_rdtsc()) return 0;
    __asm__ __volatile__("rdtsc;": "=a" (reg_a), "=d" (reg_d));
    return (reg_d<<32)|(reg_a&0xffffffffULL);
}

#endif

/** 32 Bit implementations */
#if defined(_32_BIT)

static void x86_cpuid(unsigned long long *a, unsigned long long *b, unsigned long long *c, unsigned long long *d)
{
    unsigned int reg_a,reg_b,reg_c,reg_d;

    __asm__ __volatile__(
        "cpuid;"
        : "=a" (reg_a), "=b" (reg_b), "=c" (reg_c), "=d" (reg_d)
        : "a" ((int)*a), "b" ((int)*b), "c" ((int)*c), "d" ((int)*d)
    );
    *a=(unsigned long long)reg_a;
    *b=(unsigned long long)reg_b;
    *c=(unsigned long long)reg_c;
    *d=(unsigned long long)reg_d;
}

static int x86_has_cpuid()
{
    int flags_old,flags_new;

    __asm__ __volatile__(
        "pushfl;"
        "popl %%eax;"
        : "=a" (flags_old)
    );

    flags_new=flags_old;
    if (flags_old&(1<<21)) flags_new&=0xffdfffff;
    else flags_new|=(1<<21);

    __asm__ __volatile__(
        "pushl %%eax;"
        "popfl;"
        "pushfl;"
        "popl %%eax;"
        : "=a" (flags_new)
        : "a" (flags_new)
    );

    // CPUID is supported if Bit 21 in the EFLAGS register can be changed
    if (flags_new==flags_old) return 0;
    else
    {
        __asm__ __volatile__(
            "pushl %%eax;"
            "popfl;"
            :
            : "a" (flags_old)
        );
        return 1;
    }
}

unsigned long long x86_timestamp()
{
    unsigned int reg_a,reg_d;

    if (!x86_has_rdtsc()) return 0;
    __asm__ __volatile__("rdtsc;": "=a" (reg_a) , "=d" (reg_d));
    // upper 32 Bit in EDX, lower 32 Bit in EAX
    return (((unsigned long long)reg_d)<<32)+reg_a;
}

#endif

int x86_has_rdtsc()
{
    unsigned long long a=0,b=0,c=0,d=0;

    if (!x86_has_cpuid()) return 0;

    a=0;
    x86_cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        x86_cpuid(&a,&b,&c,&d);
        if ((int)d&(1<<4)) return 1;
    }

    return 0;
}

int x86_has_invariant_rdtsc(const char *vendor)
{
    unsigned long long a=0,b=0,c=0,d=0;
    int res=0;

    if (x86_has_rdtsc()) {

        /* TSCs are usable if CPU supports only one frequency in C0 (no speedstep/Cool'n'Quite)
           or if multiple frequencies are available and the constant/invariant TSC feature flag is set */

				if (!strcmp(vendor, "GenuineIntel")) {
            /*check if Powermanagement and invariant TSC are supported*/
            if (x86_has_cpuid())
            {
                a=1;
                x86_cpuid(&a,&b,&c,&d);
                /* no Frequency control */
                if ((!(d&(1<<22)))&&(!(c&(1<<7)))) res=1;
                a=0x80000000;
                x86_cpuid(&a,&b,&c,&d);
                if (a >=0x80000007)
                {
                    a=0x80000007;
                    x86_cpuid(&a,&b,&c,&d);
                    /* invariant TSC */
                    if (d&(1<<8)) res =1;
                }
            }
        }

				if (!strcmp(vendor, "AuthenticAMD")) {
            /*check if Powermanagement and invariant TSC are supported*/
            if (x86_has_cpuid())
            {
                a=0x80000000;
                x86_cpuid(&a,&b,&c,&d);
                if (a >=0x80000007)
                {
                    a=0x80000007;
                    x86_cpuid(&a,&b,&c,&d);

                    /* no Frequency control */
                    if ((!(d&(1<<7)))&&(!(d&(1<<1)))) res=1;
                    /* invariant TSC */
                    if (d&(1<<8)) res =1;
                }
                /* assuming no frequency control if cpuid does not provide the extended function to test for it */
                else res=1;
            }
        }
    }

    return res;
}
