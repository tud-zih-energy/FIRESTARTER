/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2019 TU Dresden, Center for Information Services and High
 * Performance Computing
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

/**
 * @file x86.c
 *  architecture specific part of the hardware detection for x86 architectures
 *  Uses CPUID and RDTSC instructions if available
 *  currently only AMD and Intel CPUs are supported
 */
#include "cpu.h"
#include <stdlib.h>
#include <unistd.h>

/**
 * check if RDTSC instruction is available
 */
static int has_rdtsc();

/**
 * certain CPUs feature TSCs that are influenced by the powermanagement
 * those TSCs cannot be used to measure time
 * @return 1 if a reliable TSC exists; 0 if no TSC is available or TSC is not usable
 */
static int has_invariant_rdtsc();

//see cpu.h
#if defined (__ARCH_X86)

#if ((defined (__x86_64__))||(defined (__x86_64))||(defined (x86_64)))
#define _64_BIT
#else
#if ((defined (__i386__))||(defined (__i386))||(defined (i386))||(defined (__i486__))||(defined (__i486))||(defined (i486))||(defined (__i586__))||(defined (__i586))||(defined (i586))||(defined (__i686__))||(defined (__i686))||(defined (i686)))
#define _32_BIT
#endif
#endif

/*
 * declarations of x86 specific functions, only used within this file
 */

/**
 * check if CPUID instruction is available
 */
static int has_cpuid();

/**
 * call CPUID instruction
 */
static void cpuid(unsigned long long *a, unsigned long long *b, unsigned long long *c, unsigned long long *d);

/**
 * check if package supports more than 1 (logical) CPU
 */
static int has_htt();

/** 64 Bit implementations  */
#if defined _64_BIT

static void cpuid(unsigned long long *a, unsigned long long *b, unsigned long long *c, unsigned long long *d)
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

static int has_cpuid()
{
    // all 64 Bit x86 CPUs support CPUID
    return 1;
}

unsigned long long timestamp()
{
    unsigned long long reg_a,reg_d;

    if (!has_rdtsc()) return 0;
    __asm__ __volatile__("rdtsc;": "=a" (reg_a), "=d" (reg_d));
    return (reg_d<<32)|(reg_a&0xffffffffULL);
}

#endif

/** 32 Bit implementations */
#if defined(_32_BIT)

static void cpuid(unsigned long long *a, unsigned long long *b, unsigned long long *c, unsigned long long *d)
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

static int has_cpuid()
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

unsigned long long timestamp()
{
    unsigned int reg_a,reg_d;

    if (!has_rdtsc()) return 0;
    __asm__ __volatile__("rdtsc;": "=a" (reg_a) , "=d" (reg_d));
    // upper 32 Bit in EDX, lower 32 Bit in EAX
    return (((unsigned long long)reg_d)<<32)+reg_a;
}

#endif

/**
 * shared implementations for 32 Bit and 64 Bit mode
 */

/**
 * try to estimate ISA using compiler macros
 */
void get_architecture(char* arch, size_t len)
{
#if ((defined (__i386__))||(defined (__i386))||(defined (i386)))
    strncpy(arch,"i386",len);
#endif

#if ((defined (__i486__))||(defined (__i486))||(defined (i486)))
    strncpy(arch,"i486",len);
#endif

#if ((defined (__i586__))||(defined (__i586))||(defined (i586)))
    strncpy(arch,"i586",len);
#endif

#if ((defined (__i686__))||(defined (__i686))||(defined (i686)))
    strncpy(arch,"i686",len);
#endif

#if ((defined (__x86_64__))||(defined (__x86_64))||(defined (x86_64)))
    strncpy(arch,"x86_64",len);
#endif
}


int has_rdtsc()
{
    unsigned long long a=0,b=0,c=0,d=0;

    if (!has_cpuid()) return 0;

    a=0;
    cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        cpuid(&a,&b,&c,&d);
        if ((int)d&(1<<4)) return 1;
    }

    return 0;

}

int has_invariant_rdtsc()
{
    unsigned long long a=0,b=0,c=0,d=0;
    char tmp[_HW_DETECT_MAX_OUTPUT];
    int res=0;

    if ((has_rdtsc())&&(get_cpu_vendor((char*)&tmp[0],_HW_DETECT_MAX_OUTPUT)==0))
    {

        /* TSCs are usable if CPU supports only one frequency in C0 (no speedstep/Cool'n'Quite)
           or if multiple frequencies are available and the constant/invariant TSC feature flag is set */

        if (!strcmp(&tmp[0],"GenuineIntel"))
        {
            /*check if Powermanagement and invariant TSC are supported*/
            if (has_cpuid())
            {
                a=1;
                cpuid(&a,&b,&c,&d);
                /* no Frequency control */
                if ((!(d&(1<<22)))&&(!(c&(1<<7)))) res=1;
                a=0x80000000;
                cpuid(&a,&b,&c,&d);
                if (a >=0x80000007)
                {
                    a=0x80000007;
                    cpuid(&a,&b,&c,&d);
                    /* invariant TSC */
                    if (d&(1<<8)) res =1;
                }
            }
        }

        if (!strcmp(&tmp[0],"AuthenticAMD"))
        {
            /*check if Powermanagement and invariant TSC are supported*/
            if (has_cpuid())
            {
                a=0x80000000;
                cpuid(&a,&b,&c,&d);
                if (a >=0x80000007)
                {
                    a=0x80000007;
                    cpuid(&a,&b,&c,&d);

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

static int has_htt()
{
    unsigned long long a=0,b=0,c=0,d=0;

    if (!has_cpuid()) return 0;
    a=0;
    cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        cpuid(&a,&b,&c,&d);
        if (d&(1<<28)) return 1;
    }
    return 0;
}

int get_cpu_vendor(char* vendor, size_t len)
{
    unsigned long long a=0,b=0,c=0,d=0;
    char tmp_vendor[13];

    if (!has_cpuid()) return generic_get_cpu_vendor(vendor);
    a=0;
    cpuid(&a,&b,&c,&d);

    tmp_vendor[0]  = (char) ( b & 0xff);
    tmp_vendor[1]  = (char) ((b >> 8) & 0xff);
    tmp_vendor[2]  = (char) ((b >> 16) & 0xff);
    tmp_vendor[3]  = (char) ((b >> 24) & 0xff);
    tmp_vendor[4]  = (char) ( d & 0xff);
    tmp_vendor[5]  = (char) ((d >> 8) & 0xff);
    tmp_vendor[6]  = (char) ((d >> 16) & 0xff);
    tmp_vendor[7]  = (char) ((d >> 24) & 0xff);
    tmp_vendor[8]  = (char) ( c & 0xff);
    tmp_vendor[9]  = (char) ((c >> 8) & 0xff);
    tmp_vendor[10] = (char) ((c >> 16) & 0xff);
    tmp_vendor[11] = (char) ((c >> 24) & 0xff);
    tmp_vendor[12]='\0';

    strncpy(vendor,tmp_vendor,len);

    return 0;
}

int get_cpu_name(char* name, size_t len)
{
    unsigned long long a=0,b=0,c=0,d=0;
    char tmp[48];
    char* start;

    if (!has_cpuid()) return generic_get_cpu_name(name);
    a=0x80000000;
    cpuid(&a,&b,&c,&d);
    /* read the name string returned by cpuid */
    if (a >=0x80000004)
    {
        a=0x80000002;
        cpuid(&a,&b,&c,&d);

        tmp[0]  = (char) ( a & 0xff);
        tmp[1]  = (char) ((a >> 8) & 0xff);
        tmp[2]  = (char) ((a >> 16) & 0xff);
        tmp[3]  = (char) ((a >> 24) & 0xff);
        tmp[4]  = (char) ( b & 0xff);
        tmp[5]  = (char) ((b >> 8) & 0xff);
        tmp[6]  = (char) ((b >> 16) & 0xff);
        tmp[7]  = (char) ((b >> 24) & 0xff);
        tmp[8]  = (char) ( c & 0xff);
        tmp[9]  = (char) ((c >> 8) & 0xff);
        tmp[10] = (char) ((c >> 16) & 0xff);
        tmp[11] = (char) ((c >> 24) & 0xff);
        tmp[12] = (char) ( d & 0xff);
        tmp[13] = (char) ((d >> 8) & 0xff);
        tmp[14] = (char) ((d >> 16) & 0xff);
        tmp[15] = (char) ((d >> 24) & 0xff);

        a=0x80000003;
        cpuid(&a,&b,&c,&d);

        tmp[16] = (char) ( a & 0xff);
        tmp[17] = (char) ((a >> 8) & 0xff);
        tmp[18] = (char) ((a >> 16) & 0xff);
        tmp[19] = (char) ((a >> 24) & 0xff);
        tmp[20] = (char) ( b & 0xff);
        tmp[21] = (char) ((b >> 8) & 0xff);
        tmp[22] = (char) ((b >> 16) & 0xff);
        tmp[23] = (char) ((b >> 24) & 0xff);
        tmp[24] = (char) ( c & 0xff);
        tmp[25] = (char) ((c >> 8) & 0xff);
        tmp[26] = (char) ((c >> 16) & 0xff);
        tmp[27] = (char) ((c >> 24) & 0xff);
        tmp[28] = (char) ( d & 0xff);
        tmp[29] = (char) ((d >> 8) & 0xff);
        tmp[30] = (char) ((d >> 16) & 0xff);
        tmp[31] = (char) ((d >> 24) & 0xff);

        a=0x80000004;
        cpuid(&a,&b,&c,&d);

        tmp[32] = (char) ( a & 0xff);
        tmp[33] = (char) ((a >> 8) & 0xff);
        tmp[34] = (char) ((a >> 16) & 0xff);
        tmp[35] = (char) ((a >> 24) & 0xff);
        tmp[36] = (char) ( b & 0xff);
        tmp[37] = (char) ((b >> 8) & 0xff);
        tmp[38] = (char) ((b >> 16) & 0xff);
        tmp[39] = (char) ((b >> 24) & 0xff);
        tmp[40] = (char) ( c & 0xff);
        tmp[41] = (char) ((c >> 8) & 0xff);
        tmp[42] = (char) ((c >> 16) & 0xff);
        tmp[43] = (char) ((c >> 24) & 0xff);
        tmp[44] = (char) ( d & 0xff);
        tmp[45] = (char) ((d >> 8) & 0xff);
        tmp[46] = (char) ((d >> 16) & 0xff);
        tmp[47]='\0';

        /* remove leading whitespace */
        start=&tmp[0];
        while (*start==' ') start++;

        if (len>48) len=48;
        memset(name,0,len);
        if (len>strlen(start)) len=strlen(start)+1;
        strncpy(name,start,len);

        return 0;
    }

    return generic_get_cpu_name(name);
}

int get_cpu_family()
{
    unsigned long long a=0,b=0,c=0,d=0;

    if (!has_cpuid()) return generic_get_cpu_family();
    a=0;
    cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        cpuid(&a,&b,&c,&d);

        return (((int)a>>8)&0xf)+(((int)a>>20)&0xff);
    }
    return generic_get_cpu_family();
}
int get_cpu_model()
{
    unsigned long long a=0,b=0,c=0,d=0;

    if (!has_cpuid()) return generic_get_cpu_model();
    a=0;
    cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        cpuid(&a,&b,&c,&d);

        return (((int)a>>4)&0xf)+(((int)a>>12)&0xf0);
    }
    return generic_get_cpu_model();
}
int get_cpu_stepping()
{
    unsigned long long a=0,b=0,c=0,d=0;

    if (!has_cpuid()) return generic_get_cpu_stepping();
    a=0;
    cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        cpuid(&a,&b,&c,&d);

        return ((int)a&0xf);
    }
    return generic_get_cpu_stepping();
}

int get_cpu_isa_extensions(char* features, size_t len)
{
    unsigned long long a=0,b=0,c=0,d=0;
    unsigned long long max,max_ext;
    char tmp[16];

    if (!has_cpuid()) return generic_get_cpu_isa_extensions(features,len);

    memset(features,0,len);

    a=0;
    cpuid(&a,&b,&c,&d);
    max=a;

    a=0x80000000;
    cpuid(&a,&b,&c,&d);
    max_ext=a;

    get_cpu_vendor(tmp,sizeof(tmp));

    //identical on Intel an AMD
    if ((!strcmp("AuthenticAMD",&tmp[0]))||(!strcmp("GenuineIntel",&tmp[0])))
    {
        if (max>=1)
        {
            a=1;
            cpuid(&a,&b,&c,&d);

            if (d&(1<<4)) strncat(features,"FPU ", (len-strlen(features))-1);
            /* supported by hardware, but not usable in 64 Bit Mode */
#if defined _32_BIT
            if (d&(1<<3))  strncat(features,"PSE ",   (len-strlen(features))-1);
            if (d&(1<<6))  strncat(features,"PAE ",   (len-strlen(features))-1);
            if (d&(1<<17)) strncat(features,"PSE36 ", (len-strlen(features))-1);
#endif
            if (d&(1<<23)) strncat(features,"MMX ",   (len-strlen(features))-1);
            if (d&(1<<25)) strncat(features,"SSE ",   (len-strlen(features))-1);
            if (d&(1<<26)) strncat(features,"SSE2 ",  (len-strlen(features))-1);
            if (c&1)       strncat(features,"SSE3 ",  (len-strlen(features))-1);
            if (c&(1<<9))  strncat(features,"SSSE3 ", (len-strlen(features))-1);
            if (c&(1<<19)) strncat(features,"SSE4.1 ",(len-strlen(features))-1);
            if (c&(1<<20)) strncat(features,"SSE4.2 ",(len-strlen(features))-1);
            if (c&(1<<28)) strncat(features,"AVX ",   (len-strlen(features))-1);
            if (c&(1<<12)) strncat(features,"FMA ",   (len-strlen(features))-1);
            if (c&(1<<25)) strncat(features,"AES ",   (len-strlen(features))-1);
            if (d&(1<<8))  strncat(features,"CX8 ",   (len-strlen(features))-1);
            if (c&(1<<13)) strncat(features,"CX16 ",  (len-strlen(features))-1);
            if (c&(1<<23)) strncat(features,"POPCNT ",(len-strlen(features))-1);

        }
        if (max>=7)
        {
            a=7;c=0;
            cpuid(&a,&b,&c,&d);
            
            if (b&(1<<5))  strncat(features,"AVX2 ", (len-strlen(features))-1);
            if (b&(1<<16)) strncat(features,"AVX512 ", (len-strlen(features))-1);
        }
        if (max_ext>=0x80000001)
        {
            a=0x80000001;
            cpuid(&a,&b,&c,&d);

            #if defined _64_BIT
            if (d&(1<<29)) strncat(features,"X86_64 ",(len-strlen(features))-1);
            #endif
        }
    }

    if (has_cpuid()) strncat(features,"CPUID ",(len-strlen(features))-1);
    //AMD specific
    if (!strcmp("AuthenticAMD",&tmp[0]))
    {
        if (max_ext>=0x80000001)
        {
            a = 0x80000001;
            cpuid(&a, &b, &c, &d);

            if (d&(1<<22)) strncat(features,"MMX_EXT ",(len-strlen(features))-1);
            if (c&(1<<16)) strncat(features,"FMA4 ",(len-strlen(features))-1);
            if (c&(1<<6)) strncat(features,"SSE4A ",(len-strlen(features))-1);
            if (c&(1<<5)) strncat(features,"ABM ",(len-strlen(features))-1);
        }

        if (max_ext >= 0x80000007)
        {
            a = 0x80000007;
            cpuid(&a, &b, &c, &d);

            if ((d&(1<<7)) || (d&(1<<1)))
            {
                /* cpu supports frequency scaling
                   NOTE this is not included into the feature list, as it can't be determined with cpuid if it is actually used
                        instead sysfs is used to determine scaling policy */
            }
        }
    }
    //Intel specific
    if (!strcmp("GenuineIntel", &tmp[0]))
    {
        if (max >= 1)
        {
            a = 1;
            cpuid(&a, &b, &c, &d);
            if (c&(1<<9))
            {
                /* cpu supports frequency scaling
                   NOTE this is not included into the feature list, as it can't be determined with cpuid if it is actually used
                        instead sysfs is used to determine scaling policy */
            }
        }
    }
    //TODO other vendors
    if ((strcmp("AuthenticAMD",&tmp[0]))&&(strcmp("GenuineIntel",&tmp[0]))) return generic_get_cpu_isa_extensions(features,len);

    if (num_threads_per_core()>1) strncat(features,"SMT ",(len-strlen(features))-1);

    return 0;
}

/**
 * measures clockrate using the Time-Stamp-Counter
 * @param check if set to 1 only constant TSCs will be used (i.e. power management independent TSCs)
 *              if set to 0 non constant TSCs are allowed (e.g. AMD K8)
 * @param cpu the cpu that should be used, only relevant for the fallback to generic functions
 *            if TSC is available and check is passed or deactivated then it is assumed thet the affinity
 *            has already being set to the desired cpu
 * @return frequency in highest P-State, 0 if no invariant TSC is available
 */
unsigned long long get_cpu_clockrate(int check,int cpu)
{
    unsigned long long start1_tsc,start2_tsc,end1_tsc,end2_tsc;
    unsigned long long start_time,end_time;
    unsigned long long clock_lower_bound,clock_upper_bound,clock;
    unsigned long long clockrate=0;
    int i,num_measurements=0,min_measurements;
    char tmp[_HW_DETECT_MAX_OUTPUT];
    struct timeval ts;

    if (check)
    {
        /* non invariant TSCs can be used if CPUs run at fixed frequency */
        scaling_governor(-1, tmp, _HW_DETECT_MAX_OUTPUT);
        if (!has_invariant_rdtsc()&&(strcmp(tmp,"performance"))&&(strcmp(tmp,"powersave"))) return generic_get_cpu_clockrate(cpu);
        min_measurements=5;
    }
    else min_measurements=20;

    if (!has_rdtsc()) return generic_get_cpu_clockrate(cpu);

    i=3;
    do
    {
        //start timestamp
        start1_tsc=timestamp();
        gettimeofday(&ts,NULL);
        start2_tsc=timestamp();

        start_time=ts.tv_sec*1000000+ts.tv_usec;

        //waiting
        do {
            end1_tsc=timestamp();
        }
        while (end1_tsc<start2_tsc+1000000*i);   /* busy waiting */

        //end timestamp
        do{
          end1_tsc=timestamp();
          gettimeofday(&ts,NULL);
          end2_tsc=timestamp();
          end_time=ts.tv_sec*1000000+ts.tv_usec;
        }
        while (end_time == start_time);

        clock_lower_bound=(((end1_tsc-start2_tsc)*1000000)/(end_time-start_time));
        clock_upper_bound=(((end2_tsc-start1_tsc)*1000000)/(end_time-start_time));

        // if both values differ significantly, the measurement could have been interrupted between 2 rdtsc's
        if (((double)clock_lower_bound>(((double)clock_upper_bound)*0.999))&&((end_time-start_time)>2000))
        {
            num_measurements++;
            clock=(clock_lower_bound+clock_upper_bound)/2;
            if(clockrate==0) clockrate=clock;
            else if ((check)&&(clock<clockrate)) clockrate=clock;
            else if ((!check)&&(clock>clockrate)) clockrate=clock;
        }
        i+=2;
    }
    while (((end_time-start_time)<10000)||(num_measurements<min_measurements));

    return clockrate;
}
/**
 * number of caches (of one cpu)
 * @param cpu the cpu that should be used, only relevant for the fallback to generic functions
 *            if cpuid is available it is assumed that the affinity has already been set to the desired cpu
 */
int num_caches(int cpu)
{
    unsigned long long a=0,b=0,c=0,d=0;
    unsigned long long max,max_ext;
    char tmp[16];
    int num;

    if (!has_cpuid()) return generic_num_caches(cpu);

    a=0;
    cpuid(&a,&b,&c,&d);
    max=a;

    a=0x80000000;
    cpuid(&a,&b,&c,&d);
    max_ext=a;

    get_cpu_vendor(&tmp[0],16);

    //AMD specific
    if (!strcmp("AuthenticAMD",&tmp[0]))
    {
        if (max_ext<0x80000006) return generic_num_caches(cpu);

        a=0x80000006;
        cpuid(&a,&b,&c,&d);

        if (((c>>16)==0)||(((c>>12)&0xf)==0)) return 2; /* only L1I and L1D */
        else if (((d>>18)==0)||(((d>>12)&0xf)==0)) return 3; /* L1I, L1D, and L2 */
        else return 4; /* L1I, L1D, L2, and L3 */
    }

    //Intel specific
    if (!strcmp("GenuineIntel",&tmp[0]))
    {
        num=0;
        if (max>=0x00000004)
        {
            do
            {
                a=0x00000004;
                c=(unsigned long long)num;
                cpuid(&a,&b,&c,&d);

                num++;
            }
            while (a&0x1f);
        }
        else if (max>=0x00000002)
        {
            //TODO use function 02h if 04h is not supported
            return generic_num_caches(cpu);
        }

        return num-1;
    }

    //TODO other vendors

    return generic_num_caches(cpu);
}

/**
 * information about the cache: level, associativity...
 * @param cpu the cpu that should be used, only relevant for the fallback to generic functions
 *            if cpuid is available it is assumed that the affinity has already been set to the desired cpu
 * @param id id of the cache 0 <= id <= num_caches()-1
 * @param output preallocated buffer for the result string
 */
//TODO use sysfs if available to determine cache sharing
int cache_info(int cpu,int id, char* output, size_t len)
{
    unsigned long long a=0,b=0,c=0,d=0;
    unsigned long long max,max_ext;
    char tmp[16];

    int num;

    int size,assoc,shared,level;

    if (!has_cpuid()) return generic_cache_info(cpu,id,output,len);

    if ((num_caches(cpu)!=-1)&&(id>=num_caches(cpu))) return -1;

    memset(output,0,len);

    a=0;
    cpuid(&a,&b,&c,&d);
    max=a;

    a=0x80000000;
    cpuid(&a,&b,&c,&d);
    max_ext=a;

    get_cpu_vendor(&tmp[0],16);

    //AMD specific
    if ((!strcmp("AuthenticAMD",&tmp[0]))&&(max_ext>=0x80000005))
    {
        if (id==1)
        {
            a=0x80000005;
            cpuid(&a,&b,&c,&d);

            size=(d>>24);
            assoc=(d>>16)&0xff;

            if (assoc==0) return -1;
            else if (assoc==0x1)
                snprintf(output,len,"Level 1 Instruction Cache, %i KiB, direct mapped, per thread",size);
            else if (assoc==0xff)
                snprintf(output,len,"Level 1 Instruction Cache, %i KiB, fully associative, per thread",size);
            else
                snprintf(output,len,"Level 1 Instruction Cache, %i KiB, %i-way set associative, per thread",size,assoc);

            return 0;
        }
        if (id==0)
        {
            a=0x80000005;
            cpuid(&a,&b,&c,&d);

            size=(c>>24);
            assoc=(c>>16)&0xff;

            if (assoc==0) return -1;
            else if (assoc==0x1)
                snprintf(output,len,"Level 1 Data Cache, %i KiB, direct mapped, per thread",size);
            else if (assoc==0xff)
                snprintf(output,len,"Level 1 Date Cache, %i KiB, fully associative, per thread",size);
            else
                snprintf(output,len,"Level 1 Data Cache, %i KiB, %i-way set associative, per thread",size,assoc);

            return 0;
        }
    }
    //AMD specific
    if ((!strcmp("AuthenticAMD",&tmp[0]))&&(max_ext>=0x80000006))
    {
        if (id==2)
        {
            a=0x80000006;
            cpuid(&a,&b,&c,&d);

            size=(c>>16);
            assoc=(c>>12)&0xff;

            switch (assoc)
            {
            case 0x0:
                size=0;
                assoc=0;
                break; /* disabled */
            case 0x6:
                assoc=8;
                break;
            case 0x8:
                assoc=16;
                break;
            case 0xa:
                assoc=32;
                break;
            case 0xb:
                assoc=48;
                break;
            case 0xc:
                assoc=64;
                break;
            case 0xd:
                assoc=96;
                break;
            case 0xe:
                assoc=128;
                break;
            }

            if (assoc==0)
                snprintf(output,len,"L2 Cache disabled");
            else if (assoc==0x1)
                snprintf(output,len,"Unified Level 2 Cache, %i KiB, direct mapped, per thread",size);
            else if (assoc==0xf)
                snprintf(output,len,"Unified Level 2 Cache, %i KiB, fully associative, per thread",size);
            else
                snprintf(output,len,"Unified Level 2 Cache, %i KiB, %i-way set associative, per thread",size,assoc);

            return 0;
        }
        if (id==3)
        {
            a=0x80000006;
            cpuid(&a,&b,&c,&d);

            size=(d>>18)*512;
            assoc=(d>>12)&0xff;
            //TODO 12-core MCM ???
            shared=num_cores_per_package();

            switch (assoc)
            {
            case 0x0:
                size=0;
                assoc=0;
                break; /* disabled */
            case 0x6:
                assoc=8;
                break;
            case 0x8:
                assoc=16;
                break;
            case 0xa:
                assoc=32;
                break;
            case 0xb:
                assoc=48;
                break;
            case 0xc:
                assoc=64;
                break;
            case 0xd:
                assoc=96;
                break;
            case 0xe:
                assoc=128;
                break;
            }

            if (assoc==0)
                snprintf(output,len,"L3 Cache disabled");
            else if (assoc==0x1)
                snprintf(output,len,"Unified Level 3 Cache, %i KiB, direct mapped, shared among %i threads",size,shared);
            else if (assoc==0xf)
                snprintf(output,len,"Unified Level 3 Cache, %i KiB, fully associative, shared among %i threads",size,shared);
            else
                snprintf(output,len,"Unified Level 3 Cache, %i KiB, %i-way set associative, shared among %i threads",size,assoc,shared);

            return 0;
        }
    }

    //Intel specific
    if (!strcmp("GenuineIntel",&tmp[0]))
    {
        if ((get_cpu_family()==15)&&(max>=0x00000002)) id--;
        if (id==-1)
        {
            int descriptors[15];
            int i,j,iter;

            a=0x00000002;
            cpuid(&a,&b,&c,&d);

            iter=(a&0xff);

            for (i=0; i<iter; i++)
            {
                size=0;

                a=0x00000002;
                cpuid(&a,&b,&c,&d);

                if (!(a&0x80000000))
                {
                    descriptors[0]=(a>>8)&0xff;
                    descriptors[1]=(a>>16)&0xff;
                    descriptors[2]=(a>>24)&0xff;
                }
                else
                {
                    descriptors[0]=0;
                    descriptors[1]=0;
                    descriptors[2]=0;
                }

                for (j=1; j<4; j++) descriptors[j-1]=(a>>(8*j))&0xff;
                for (j=0; j<4; j++)
                {
                    if (!(b&0x80000000)) descriptors[j+3]=(b>>(8*j))&0xff;
                    else  descriptors[j+3]=0;
                    if (!(c&0x80000000)) descriptors[j+7]=(c>>(8*j))&0xff;
                    else  descriptors[j+7]=0;
                    if (!(d&0x80000000)) descriptors[j+11]=(d>>(8*j))&0xff;
                    else  descriptors[j+11]=0;
                }
                for (j=0; j<15; j++)
                {
                    switch(descriptors[j])
                    {
                    case 0x00:
                        break;
                    case 0x70:
                        size=12;
                        assoc=8;
                        break;
                    case 0x71:
                        size=16;
                        assoc=8;
                        break;
                    case 0x72:
                        size=32;
                        assoc=8;
                        break;
                    case 0x73:
                        size=64;
                        assoc=8;
                        break;
                    }
                    if(size)
                    {
                        shared=num_threads_per_core();
                        if (shared>1)
                            snprintf(output,len,"Level 1 Instruction Trace Cache, %i K Microops, %i-way set associative, shared among %i threads",size,assoc,shared);
                        else
                            snprintf(output,len,"Level 1 Instruction Trace Cache, %i K Microops, %i-way set associative, per thread",size,assoc);
                    }
                }

            }
        }
        else if (max>=0x00000004)
        {
            int type;
            num=0;
            do
            {
                a=0x00000004;
                c=(unsigned long long)num;
                cpuid(&a,&b,&c,&d);
                num++;
            }
            while (num<=id);

            level=((a&0xe0)>>5);
            shared=((a&0x03ffc000)>>14)+1;
            size=((((b&0xffc00000)>>22)+1)*(((b&0x3ff000)>>12)+1)*((b&0x0fff)+1)*(c+1))/1024;
            if (a&0x200) assoc=0;
            else assoc=((b&0xffc00000)>>22)+1;
            type=(a&0x1f);

            /* Hyperthreading, Netburst*/
            if (get_cpu_family()==15) shared=num_threads_per_core();
            /* Hyperthreading, Nehalem/Atom */
            /* TODO check if there are any models that do not work with that */
            if ((get_cpu_family()==6)&&(get_cpu_model()>=26))
            {
                if (level<3) shared=num_threads_per_core();
                if (level==3) shared=num_threads_per_package();
            }

            if (type==2)
            {
                if (assoc)
                {
                    if (shared>1) snprintf(output,len,"Level %i Instruction Cache, %i KiB, %i-way set associative, shared among %i threads",level,size,assoc,shared);
                    else snprintf(output,len,"Level %i Instruction Cache, %i KiB, %i-way set associative, per thread",level,size,assoc);
                }
                else
                {
                    if (shared>1) snprintf(output,len,"Level %i Instruction Cache, %i KiB, fully associative, shared among %i threads",level,size,shared);
                    else snprintf(output,len,"Level %i Instruction Cache, %i KiB, fully associative, per thread",level,size);
                }
            }
            if (type==1)
            {
                if (assoc)
                {
                    if (shared>1) snprintf(output,len,"Level %i Data Cache, %i KiB, %i-way set associative, shared among %i threads",level,size,assoc,shared);
                    else snprintf(output,len,"Level %i Date Cache, %i KiB, %i-way set associative, per thread",level,size,assoc);
                }
                else
                {
                    if (shared>1) snprintf(output,len,"Level %i Data Cache, %i KiB, fully associative, shared among %i threads",level,size,shared);
                    else snprintf(output,len,"Level %i Data Cache, %i KiB, fully associative, per thread",level,size);
                }
            }
            if (type==3)
            {
                if (assoc)
                {
                    if (shared>1) snprintf(output,len,"Unified Level %i Cache, %i KiB, %i-way set associative, shared among %i threads",level,size,assoc,shared);
                    else snprintf(output,len,"Unified Level %i Cache, %i KiB, %i-way set associative, per thread",level,size,assoc);
                }
                else
                {
                    if (shared>1) snprintf(output,len,"Unified Level %i Cache, %i KiB, fully associative, shared among %i threads",level,size,shared);
                    else snprintf(output,len,"Unified Level %i Cache, %i KiB, fully associative, per thread",level,size);
                }
            }
        }
        else if (max>=0x00000002)
        {
            //TODO use function 02h if 04h is not supported
            return generic_cache_info(cpu,id,output,len);
        }

        return 0;
    }
    //TODO other vendors

    return generic_cache_info(cpu,id,output,len);
}

int cache_level(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    cache_info(cpu,id,tmp,sizeof(tmp));
    beg=strstr(tmp,"Level");
    if (beg==NULL) return generic_cache_level(cpu,id);
    else beg+=6;
    end=strstr(beg," ");
    if (end!=NULL)*end='\0';

    return atoi(beg);
}
unsigned long long cache_size(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    cache_info(cpu,id,tmp,sizeof(tmp));
    beg=strstr(tmp,",");
    if (beg==NULL) return generic_cache_size(cpu,id);
    else beg+=2;
    end=strstr(beg,",");
    if (end!=NULL) *end='\0';
    end=strstr(beg,"KiB");
    if (end!=NULL)
    {
        end--;
        *end='\0';
        return atoi(beg)*1024;
    }
    end=strstr(beg,"MiB");
    if (end!=NULL)
    {
        end--;
        *end='\0';
        return atoi(beg)*1024*1024;
    }

    return generic_cache_size(cpu,id);
}
unsigned int cache_assoc(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    cache_info(cpu,id,tmp,sizeof(tmp));
    beg=strstr(tmp,",")+1;
    if (beg==NULL) return generic_cache_assoc(cpu,id);
    else beg++;
    end=strstr(beg,",")+1;
    if (end==NULL) return generic_cache_assoc(cpu,id);
    else end++;
    beg=end;
    end=strstr(beg,",");
    if (end!=NULL) *end='\0';
    end=strstr(tmp,"-way");
    if (end!=NULL) {
        *end='\0';
        return atoi(beg);
    }
    end=strstr(tmp,"fully");
    if (end!=NULL) {
        *end='\0';
        return FULLY_ASSOCIATIVE;
    }
    return generic_cache_assoc(cpu,id);
}
int cache_type(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    cache_info(cpu,id,tmp,sizeof(tmp));
    beg=tmp;
    end=strstr(beg,",");
    if (end!=NULL)*end='\0';
    else return generic_cache_type(cpu,id);

    if (strstr(beg,"Unified")!=NULL) return UNIFIED_CACHE;
    if (strstr(beg,"Trace")!=NULL) return INSTRUCTION_TRACE_CACHE;
    if (strstr(beg,"Data")!=NULL) return DATA_CACHE;
    if (strstr(beg,"Instruction")!=NULL) return INSTRUCTION_CACHE;

    return generic_cache_type(cpu,id);
}
int cache_shared(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    cache_info(cpu,id,tmp,sizeof(tmp));
    beg=strstr(tmp,"among");
    if (beg==NULL)
    {
        beg=strstr(tmp,"per thread");
        if (beg!=NULL) return 1;
        else return generic_cache_shared(cpu,id);
    }
    beg+=6;
    end=strstr(beg,"thread");
    if (end!=NULL)*(end--)='\0';

    return atoi(beg);
}
int cacheline_length(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    cache_info(cpu,id,tmp,sizeof(tmp));
    beg=strstr(tmp,",")+1;
    if (beg==NULL) return generic_cacheline_length(cpu,id);
    else beg++;
    end=strstr(beg,",")+1;
    if (end==NULL) return generic_cacheline_length(cpu,id);
    else end++;
    beg=end;
    end=strstr(beg,",")+1;
    if (end==NULL) return generic_cacheline_length(cpu,id);
    else end++;
    beg=end;
    end=strstr(beg,"Byte cachelines");
    if (end!=NULL) *(end--)='\0';

    return atoi(beg);
}

int num_packages()
{
    if ((num_cpus()==-1)||(num_threads_per_package()==-1)) return generic_num_packages();
    else if (!has_htt()) return num_cpus();
    else return num_cpus()/num_threads_per_package();
}

int num_cores_per_package()
{
    unsigned long long a=0,b=0,c=0,d=0;
    char tmp[16];
    int num=-1;

    if (!has_htt()) return 1;
    if (get_cpu_vendor(tmp,16)!=0) return generic_num_cores_per_package();

    if (!strcmp(&tmp[0],"GenuineIntel"))
    {
        /* prefer generic implementation on Processors that might support Hyperthreading */
        /* TODO check if there are any models above 26 that don't have HT*/
        if (generic_num_cores_per_package()!=-1)
        {
            /* Hyperthreading, Netburst*/
            if (get_cpu_family()==15) num=generic_num_cores_per_package();
            /* Hyperthreading, Nehalem/Atom*/
            if ((get_cpu_family()==6)&&(get_cpu_model()>=26)) num=generic_num_cores_per_package();
            if (num!=-1) return num;
        }

        a=0;
        cpuid(&a,&b,&c,&d);
        if (a>=4)
        {
            a=4;
            c=0;
            cpuid(&a,&b,&c,&d);
            num= ( a >> 26 ) + 1 ;
        }
        else num=1;

        if (num>num_cpus()) num=num_cpus();
        return num;
    }
    if (!strcmp(&tmp[0],"AuthenticAMD"))
    {
        a=0x80000000;
        cpuid(&a,&b,&c,&d);
        if (a>=0x80000008)
        {
            a=0x80000008;
            cpuid(&a,&b,&c,&d);
            num= (c&0xff)+1;

            if (get_cpu_family() >= 0x17)
            {
                a=0x8000001e;
                cpuid(&a,&b,&c,&d);
                num/=(b>>8&0xff)+1;
            }
        }
        else num=1;
        /* consistency checks */
        /* more cores than cpus is not possible -> some cores are deactivated */
        if (num>num_cpus()) num=num_cpus();
        /* if the number of packages is known this can be checked for multi-socket systems, too
           NOTE depends on valid entries in sysfs */
        if ((generic_num_packages()!=-1)&&(generic_num_packages()*num>num_cpus())) num=num_cpus()/generic_num_packages();

        return num;
    }
    //TODO other vendors

    return generic_num_cores_per_package();
}

int num_threads_per_core()
{
    return num_threads_per_package()/num_cores_per_package();
}

int num_threads_per_package()
{
    unsigned long long a=0,b=0,c=0,d=0;
    int num=-1;
    char tmp[16];

    if (has_cpuid())
    {
        if (!has_htt()) return 1;
        get_cpu_vendor(tmp,16);
        a=0;
        cpuid(&a,&b,&c,&d);
        if (a>=1)
        {
            /* prefer generic implementation on Processors that support Hyperthreading */
            /* TODO check if there are any models above 26 that don't have HT*/
            if ((!strcmp(tmp,"GenuineIntel")) && (generic_num_threads_per_package() != -1))
            {
                /* Hyperthreading, Netburst*/
                if (get_cpu_family() == 15) num = generic_num_threads_per_package();
                /* Hyperthreading, Nehalem/Atom */
                if ((get_cpu_family() == 6) && (get_cpu_model() >= 26)) num = generic_num_threads_per_package();
                if (num != -1) return num;
            }

            a = 1;
            cpuid(&a, &b, &c, &d);
            num = ((b>>16)&0xff);

            /* check if SMT is supported but deactivated (cpuid reports maximum logical processor count, even if some are deactivated in BIOS) */
            /* this simple test will do the trick for single socket systems (e.g. Pentium 4/D) */
            if (num > num_cpus()) num = num_cpus();
            /* distinguishing between a dual socket system that supports but does not use SMT and a single socket system that uses SMT
               is not as trivial:
               e.g. dual socket single core Xeon with deactivated Hyperthreading vs. single socket single core Xeon with enabled HT
                    -> - num_cpus = 2 (reported by sysconf)
                       - num_threads_per_package = 2 (cpuid.1:EBX[23:16] -> maximal logical processor count)
                       - num_cores_per_package = 1  (cpuid.4:EAX[31:26]+1)
            NOTE if sysfs/cpuinfo detection of physical packages fails the dual socket system with deactivated
                 Hyperthreading will be reported as single socket system with enabled HyperThreading */
            if ((generic_num_packages() != -1) && (generic_num_packages()*num > num_cpus())) num = num_cpus() / generic_num_packages();

            return num;
        }
        else if (generic_num_threads_per_package()!=-1) return generic_num_threads_per_package();
        else return 1;
    }
    else if (generic_num_threads_per_package()!=-1) return generic_num_threads_per_package();
    else return 1;
}

#endif

