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
 * @file x86_win64.c
 *  simplified hardware detection for x86 architectures under 64 bit windows
 */

#include "cpu.h"
#include <stdlib.h>
#include <unistd.h>

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

int get_cpu_vendor(char* vendor, size_t len)
{
    unsigned long long a=0,b=0,c=0,d=0;
    char tmp_vendor[13];

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

    return -1;
}

unsigned long long timestamp()
{
    unsigned long long reg_a,reg_d;

    __asm__ __volatile__("rdtsc;": "=a" (reg_a), "=d" (reg_d));
    return (reg_d<<32)|(reg_a&0xffffffffULL);
}

/**
 * measures clockrate using the Time-Stamp-Counter
 * @param check if set to 1 only constant TSCs will be used (i.e. power management independent TSCs)
 *              if set to 0 non constant TSCs are allowed (e.g. AMD K8)
 *              TODO this is currently not supported under windows (parameter ignored, using any TSC available)
 * @param cpu the cpu that should be used, only relevant for the fallback to generic functions
 *            if TSC is available and check is passed or deactivated then it is assumed thet the affinity
 *            has already being set to the desired cpu
 * @return frequency in highest P-State, 0 if no invariant TSC is available
 */
unsigned long long get_cpu_clockrate(int check, int cpu)
{
    unsigned long long start1_tsc,start2_tsc,end1_tsc,end2_tsc;
    unsigned long long start_time,end_time;
    unsigned long long clock_lower_bound,clock_upper_bound,clock;
    unsigned long long clockrate=0;
    int i,num_measurements=0,min_measurements;
    struct timeval ts;

    min_measurements=20;

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

int get_cpu_family()
{
    unsigned long long a=0,b=0,c=0,d=0;

    a=0;
    cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        cpuid(&a,&b,&c,&d);

        return (((int)a>>8)&0xf)+(((int)a>>20)&0xff);
    }
    return -1;
}
int get_cpu_model()
{
    unsigned long long a=0,b=0,c=0,d=0;

    a=0;
    cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        cpuid(&a,&b,&c,&d);

        return (((int)a>>4)&0xf)+(((int)a>>12)&0xf0);
    }
    return -1;
}
int get_cpu_stepping()
{
    unsigned long long a=0,b=0,c=0,d=0;

    a=0;
    cpuid(&a,&b,&c,&d);
    if (a>=1)
    {
        a=1;
        cpuid(&a,&b,&c,&d);

        return ((int)a&0xf);
    }
    return -1;
}

int get_cpu_isa_extensions(char* features, size_t len)
{
    unsigned long long a=0,b=0,c=0,d=0;
    unsigned long long max,max_ext;
    char tmp[16];

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
    if ((strcmp("AuthenticAMD",&tmp[0]))&&(strcmp("GenuineIntel",&tmp[0]))) return -1;

    return 0;
}

