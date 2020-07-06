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

#ifndef __FIRESTARTER__WORK_H
#define __FIRESTARTER__WORK_H

#include "firestarter_global.h"
#include <mm_malloc.h>

/*
 * function definitions
 */
#define FUNC_KNL_XEONPHI_AVX512_4T     1
#define FUNC_SKL_COREI_FMA_1T          2
#define FUNC_SKL_COREI_FMA_2T          3
#define FUNC_SKL_XEONEP_AVX512_1T      4
#define FUNC_SKL_XEONEP_AVX512_2T      5
#define FUNC_HSW_COREI_FMA_1T          6
#define FUNC_HSW_COREI_FMA_2T          7
#define FUNC_HSW_XEONEP_FMA_1T         8
#define FUNC_HSW_XEONEP_FMA_2T         9
#define FUNC_SNB_COREI_AVX_1T          10
#define FUNC_SNB_COREI_AVX_2T          11
#define FUNC_SNB_XEONEP_AVX_1T         12
#define FUNC_SNB_XEONEP_AVX_2T         13
#define FUNC_NHM_COREI_SSE2_1T         14
#define FUNC_NHM_COREI_SSE2_2T         15
#define FUNC_NHM_XEONEP_SSE2_1T        16
#define FUNC_NHM_XEONEP_SSE2_2T        17
#define FUNC_BLD_OPTERON_FMA4_1T       18
#define FUNC_ZEN_EPYC_ZEN_FMA_1T       19
#define FUNC_ZEN_EPYC_ZEN_FMA_2T       20

/*
 * function that does the measurement
 */
extern void _work(volatile mydata_t* data, unsigned long long *high);

/*
 * loop executed by all threads, except the master thread
 */
extern void *thread(void *threaddata);

/*
 * init functions
 */
int init_knl_xeonphi_avx512_4t(threaddata_t* threaddata) __attribute__((noinline));
int init_knl_xeonphi_avx512_4t(threaddata_t* threaddata);

int init_skl_corei_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_corei_fma_1t(threaddata_t* threaddata);

int init_skl_corei_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_corei_fma_2t(threaddata_t* threaddata);

int init_skl_xeonep_avx512_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_xeonep_avx512_1t(threaddata_t* threaddata);

int init_skl_xeonep_avx512_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_skl_xeonep_avx512_2t(threaddata_t* threaddata);

int init_hsw_corei_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_hsw_corei_fma_1t(threaddata_t* threaddata);

int init_hsw_corei_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_hsw_corei_fma_2t(threaddata_t* threaddata);

int init_hsw_xeonep_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_hsw_xeonep_fma_1t(threaddata_t* threaddata);

int init_hsw_xeonep_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_hsw_xeonep_fma_2t(threaddata_t* threaddata);

int init_snb_corei_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_corei_avx_1t(threaddata_t* threaddata);

int init_snb_corei_avx_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_corei_avx_2t(threaddata_t* threaddata);

int init_snb_xeonep_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_xeonep_avx_1t(threaddata_t* threaddata);

int init_snb_xeonep_avx_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_snb_xeonep_avx_2t(threaddata_t* threaddata);

int init_nhm_corei_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_corei_sse2_1t(threaddata_t* threaddata);

int init_nhm_corei_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_corei_sse2_2t(threaddata_t* threaddata);

int init_nhm_xeonep_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_xeonep_sse2_1t(threaddata_t* threaddata);

int init_nhm_xeonep_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_xeonep_sse2_2t(threaddata_t* threaddata);

int init_bld_opteron_fma4_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_bld_opteron_fma4_1t(threaddata_t* threaddata);

int init_zen_epyc_zen_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_zen_epyc_zen_fma_1t(threaddata_t* threaddata);

int init_zen_epyc_zen_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_zen_epyc_zen_fma_2t(threaddata_t* threaddata);


/*
 * stress test functions
 */
int asm_work_knl_xeonphi_avx512_4t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_knl_xeonphi_avx512_4t(threaddata_t* threaddata);

int asm_work_skl_corei_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_skl_corei_fma_1t(threaddata_t* threaddata);

int asm_work_skl_corei_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_skl_corei_fma_2t(threaddata_t* threaddata);

int asm_work_skl_xeonep_avx512_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_skl_xeonep_avx512_1t(threaddata_t* threaddata);

int asm_work_skl_xeonep_avx512_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_skl_xeonep_avx512_2t(threaddata_t* threaddata);

int asm_work_hsw_corei_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_hsw_corei_fma_1t(threaddata_t* threaddata);

int asm_work_hsw_corei_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_hsw_corei_fma_2t(threaddata_t* threaddata);

int asm_work_hsw_xeonep_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_hsw_xeonep_fma_1t(threaddata_t* threaddata);

int asm_work_hsw_xeonep_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_hsw_xeonep_fma_2t(threaddata_t* threaddata);

int asm_work_snb_corei_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_corei_avx_1t(threaddata_t* threaddata);

int asm_work_snb_corei_avx_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_corei_avx_2t(threaddata_t* threaddata);

int asm_work_snb_xeonep_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_xeonep_avx_1t(threaddata_t* threaddata);

int asm_work_snb_xeonep_avx_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_xeonep_avx_2t(threaddata_t* threaddata);

int asm_work_nhm_corei_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_nhm_corei_sse2_1t(threaddata_t* threaddata);

int asm_work_nhm_corei_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_nhm_corei_sse2_2t(threaddata_t* threaddata);

int asm_work_nhm_xeonep_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_nhm_xeonep_sse2_1t(threaddata_t* threaddata);

int asm_work_nhm_xeonep_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_nhm_xeonep_sse2_2t(threaddata_t* threaddata);

int asm_work_bld_opteron_fma4_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_bld_opteron_fma4_1t(threaddata_t* threaddata);

int asm_work_zen_epyc_zen_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_zen_epyc_zen_fma_1t(threaddata_t* threaddata);

int asm_work_zen_epyc_zen_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_zen_epyc_zen_fma_2t(threaddata_t* threaddata);


/*
 * low load function
 */
int low_load_function(unsigned long long addrHigh,unsigned int period) __attribute__((noinline));
int low_load_function(unsigned long long addrHigh,unsigned int period);

#endif

