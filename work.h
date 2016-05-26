/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2016 TU Dresden, Center for Information Services and High
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

#ifndef __WORK_H
#define __WORK_H

#include "firestarter_global.h"
#include <mm_malloc.h>

#define FUNC_SKL_COREI_FMA_1T	1
#define FUNC_SKL_COREI_FMA_2T	2
#define FUNC_HSW_COREI_FMA_1T	3
#define FUNC_HSW_COREI_FMA_2T	4
#define FUNC_HSW_XEONEP_FMA_1T	5
#define FUNC_HSW_XEONEP_FMA_2T	6
#define FUNC_SNB_COREI_AVX_1T	7
#define FUNC_SNB_COREI_AVX_2T	8
#define FUNC_SNB_XEONEP_AVX_1T	9
#define FUNC_SNB_XEONEP_AVX_2T	10
#define FUNC_NHM_COREI_SSE2_1T	11
#define FUNC_NHM_COREI_SSE2_2T	12
#define FUNC_NHM_XEONEP_SSE2_1T	13
#define FUNC_NHM_XEONEP_SSE2_2T	14
#define FUNC_BLD_OPTERON_FMA4_1T	15


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
int init_skl_corei_fma_1t(unsigned long long addrMem) __attribute__((noinline));
int init_skl_corei_fma_1t(unsigned long long addrMem);

int init_skl_corei_fma_2t(unsigned long long addrMem) __attribute__((noinline));
int init_skl_corei_fma_2t(unsigned long long addrMem);

int init_hsw_corei_fma_1t(unsigned long long addrMem) __attribute__((noinline));
int init_hsw_corei_fma_1t(unsigned long long addrMem);

int init_hsw_corei_fma_2t(unsigned long long addrMem) __attribute__((noinline));
int init_hsw_corei_fma_2t(unsigned long long addrMem);

int init_hsw_xeonep_fma_1t(unsigned long long addrMem) __attribute__((noinline));
int init_hsw_xeonep_fma_1t(unsigned long long addrMem);

int init_hsw_xeonep_fma_2t(unsigned long long addrMem) __attribute__((noinline));
int init_hsw_xeonep_fma_2t(unsigned long long addrMem);

int init_snb_corei_avx_1t(unsigned long long addrMem) __attribute__((noinline));
int init_snb_corei_avx_1t(unsigned long long addrMem);

int init_snb_corei_avx_2t(unsigned long long addrMem) __attribute__((noinline));
int init_snb_corei_avx_2t(unsigned long long addrMem);

int init_snb_xeonep_avx_1t(unsigned long long addrMem) __attribute__((noinline));
int init_snb_xeonep_avx_1t(unsigned long long addrMem);

int init_snb_xeonep_avx_2t(unsigned long long addrMem) __attribute__((noinline));
int init_snb_xeonep_avx_2t(unsigned long long addrMem);

int init_nhm_corei_sse2_1t(unsigned long long addrMem) __attribute__((noinline));
int init_nhm_corei_sse2_1t(unsigned long long addrMem);

int init_nhm_corei_sse2_2t(unsigned long long addrMem) __attribute__((noinline));
int init_nhm_corei_sse2_2t(unsigned long long addrMem);

int init_nhm_xeonep_sse2_1t(unsigned long long addrMem) __attribute__((noinline));
int init_nhm_xeonep_sse2_1t(unsigned long long addrMem);

int init_nhm_xeonep_sse2_2t(unsigned long long addrMem) __attribute__((noinline));
int init_nhm_xeonep_sse2_2t(unsigned long long addrMem);

int init_bld_opteron_fma4_1t(unsigned long long addrMem) __attribute__((noinline));
int init_bld_opteron_fma4_1t(unsigned long long addrMem);


/*
 * stress test functions
 */
int asm_work_skl_corei_fma_1t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_skl_corei_fma_1t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_skl_corei_fma_2t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_skl_corei_fma_2t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_hsw_corei_fma_1t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_hsw_corei_fma_1t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_hsw_corei_fma_2t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_hsw_corei_fma_2t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_hsw_xeonep_fma_1t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_hsw_xeonep_fma_1t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_hsw_xeonep_fma_2t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_hsw_xeonep_fma_2t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_snb_corei_avx_1t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_snb_corei_avx_1t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_snb_corei_avx_2t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_snb_corei_avx_2t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_snb_xeonep_avx_1t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_snb_xeonep_avx_1t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_snb_xeonep_avx_2t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_snb_xeonep_avx_2t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_nhm_corei_sse2_1t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_nhm_corei_sse2_1t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_nhm_corei_sse2_2t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_nhm_corei_sse2_2t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_nhm_xeonep_sse2_1t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_nhm_xeonep_sse2_1t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_nhm_xeonep_sse2_2t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_nhm_xeonep_sse2_2t(unsigned long long addrMem, unsigned long long addrHigh);

int asm_work_bld_opteron_fma4_1t(unsigned long long addrMem, unsigned long long addrHigh) __attribute__((noinline));
int asm_work_bld_opteron_fma4_1t(unsigned long long addrMem, unsigned long long addrHigh);


/*
 * low load function
 */
int low_load_function(unsigned long long addrHigh,unsigned int period) __attribute__((noinline));
int low_load_function(unsigned long long addrHigh,unsigned int period);

#endif
