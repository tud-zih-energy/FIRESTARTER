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

#include "work.h"

int init_nhm_corei_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_corei_sse2_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 106725376 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 106725376-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=3066;
    threaddata->bytes=1344;

    return EXIT_SUCCESS;
}
int init_nhm_corei_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_corei_sse2_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 53362688 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 53362688-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=1460;
    threaddata->bytes=640;

    return EXIT_SUCCESS;
}
int init_nhm_xeonep_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_xeonep_sse2_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 107249664 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 107249664-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=3024;
    threaddata->bytes=1536;

    return EXIT_SUCCESS;
}
int init_nhm_xeonep_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_xeonep_sse2_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 53624832 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 53624832-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=1512;
    threaddata->bytes=768;

    return EXIT_SUCCESS;
}

int init_snb_corei_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_corei_avx_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 106725376 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 106725376-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=6040;
    threaddata->bytes=1280;

    return EXIT_SUCCESS;
}
int init_snb_corei_avx_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_corei_avx_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 53362688 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 53362688-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=3020;
    threaddata->bytes=640;

    return EXIT_SUCCESS;
}
int init_snb_xeonep_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_xeonep_avx_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 107773952 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 107773952-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=5940;
    threaddata->bytes=2112;

    return EXIT_SUCCESS;
}
int init_snb_xeonep_avx_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_xeonep_avx_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-10;
    for (i = INIT_BLOCKSIZE; i <= 53886976 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 53886976-8; i+=8) *((double*)(addrMem+i)) = i * 1.654738925401e-15;

    threaddata->flops=2700;
    threaddata->bytes=960;

    return EXIT_SUCCESS;
}

int init_skl_corei_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_corei_fma_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 106725376 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 106725376-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=21200;
    threaddata->bytes=1920;

    return EXIT_SUCCESS;
}
int init_skl_corei_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_corei_fma_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 53362688 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 53362688-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=10600;
    threaddata->bytes=960;

    return EXIT_SUCCESS;
}
int init_hsw_corei_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_hsw_corei_fma_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 106725376 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 106725376-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=14880;
    threaddata->bytes=1280;

    return EXIT_SUCCESS;
}
int init_hsw_corei_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_hsw_corei_fma_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 53362688 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 53362688-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=7440;
    threaddata->bytes=640;

    return EXIT_SUCCESS;
}
int init_hsw_xeonep_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_hsw_xeonep_fma_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 107773952 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 107773952-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=15648;
    threaddata->bytes=1536;

    return EXIT_SUCCESS;
}
int init_hsw_xeonep_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_hsw_xeonep_fma_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 53886976 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 53886976-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=7824;
    threaddata->bytes=768;

    return EXIT_SUCCESS;
}

int init_bld_opteron_fma4_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_bld_opteron_fma4_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 106708992 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 106708992-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=14760;
    threaddata->bytes=640;

    return EXIT_SUCCESS;
}

int init_knl_xeonphi_avx512_4t(threaddata_t* threaddata) __attribute__((noinline));
int init_knl_xeonphi_avx512_4t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 65762645 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 65762645-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=10656;
    threaddata->bytes=1152;

    return EXIT_SUCCESS;
}
int init_skl_xeonep_avx512_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_xeonep_avx512_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 1051099136 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 1051099136-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=32160;
    threaddata->bytes=1344;

    return EXIT_SUCCESS;
}
int init_skl_xeonep_avx512_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_xeonep_avx512_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 525549568 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 525549568-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=10720;
    threaddata->bytes=448;

    return EXIT_SUCCESS;
}

int init_zen_epyc_zen_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_zen_epyc_zen_fma_1t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 107544576 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 107544576-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=12040;
    threaddata->bytes=2560;

    return EXIT_SUCCESS;
}
int init_zen_epyc_zen_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_zen_epyc_zen_fma_2t(threaddata_t* threaddata)
{
    unsigned long long addrMem = threaddata->addrMem;
    int i;

    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;
    for (i = INIT_BLOCKSIZE; i <= 53772288 - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);
    for (; i <= 53772288-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;

    threaddata->flops=4816;
    threaddata->bytes=1024;

    return EXIT_SUCCESS;
}

