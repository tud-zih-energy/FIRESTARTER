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

/**
 * assembler implementation of processor and memory stress test
 * ISA: AVX
 * optimized for Sandy Bridge - 1 thread(s) per core
 */
int asm_work_snb_corei_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_corei_avx_1t(threaddata_t* threaddata)
{
    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
        /* input: 
         *   - threaddata->addrMem    -> rax
         *   - threaddata->addrHigh   -> rbx
         *   - threaddata->iterations -> rcx
         * output: 
         *   - rax -> threaddata->iterations
         * register usage:
         *   - rax:           stores original pointer to buffer, used to periodically reset other pointers
         *   - rbx:           pointer to L1 buffer
         *   - rcx:           pointer to L2 buffer
         *   - rdx:           pointer to L3 buffer
         *   - rdi:           pointer to RAM buffer
         *   - r8:            counter for L2-pointer reset
         *   - r9:            counter for L3-pointer reset
         *   - r10:           counter for RAM-pointer reset
         *   - r11:           register for temporary results
         *   - r12:           stores cacheline width as increment for buffer addresses
         *   - r13:           stores address of shared variable that controls load level
         *   - r14:           stores iteration counter
         *   - mm*,xmm*,ymm*: data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r13;" // store address of shared variable that controls load level in r13
        "mov %%rcx, %%r14;" // store iteration counter in r14
        "mov $64, %%r12;" // increment after each cache/memory access
        //Initialize AVX-Registers for Addition
        "vmovapd 0(%%rax), %%ymm0;"
        "vmovapd 32(%%rax), %%ymm1;"
        "vmovapd 64(%%rax), %%ymm2;"
        "vmovapd 96(%%rax), %%ymm3;"
        "vmovapd 128(%%rax), %%ymm4;"
        "vmovapd 160(%%rax), %%ymm5;"
        "vmovapd 192(%%rax), %%ymm6;"
        "vmovapd 224(%%rax), %%ymm7;"
        "vmovapd 256(%%rax), %%ymm8;"
        "vmovapd 288(%%rax), %%ymm9;"
        //Initialize MMX-Registers for shift operations
        "movabs $0x5555555555555555, %%r11;"
        "movq %%r11, %%mm0;"
        "movq %%mm0, %%mm1;"
        "movq %%mm0, %%mm2;"
        "movq %%mm0, %%mm3;"
        "movq %%mm0, %%mm4;"
        "movq %%mm0, %%mm5;"
        //Initialize AVX-Registers for Transfer-Operations
        "movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
        "pinsrq $0, %%r11, %%xmm10;"
        "pinsrq $1, %%r11, %%xmm10;"
        "vinsertf128 $1, %%xmm10, %%ymm10, %%ymm10;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm11;"
        "pinsrq $1, %%r11, %%xmm11;"
        "vinsertf128 $1, %%xmm11, %%ymm11, %%ymm11;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm12;"
        "pinsrq $1, %%r11, %%xmm12;"
        "vinsertf128 $1, %%xmm12, %%ymm12, %%ymm12;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm13;"
        "pinsrq $1, %%r11, %%xmm13;"
        "vinsertf128 $1, %%xmm13, %%ymm13, %%ymm13;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm14;"
        "pinsrq $1, %%r11, %%xmm14;"
        "vinsertf128 $1, %%xmm14, %%ymm14, %%ymm14;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm15;"
        "pinsrq $1, %%r11, %%xmm15;"
        "vinsertf128 $1, %%xmm15, %%ymm15, %%ymm15;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;" // address for L2-buffer
        "mov %%rax, %%rdx;"
        "add $262144, %%rdx;" // address for L3-buffer
        "mov %%rax, %%rdi;"
        "add $1572864, %%rdi;" // address for RAM-buffer
        "movabs $32, %%r8;" // reset-counter for L2-buffer with 100 cache lines accessed per loop (200.0 KB)
        "movabs $491, %%r9;" // reset-counter for L3-buffer with 40 cache lines accessed per loop (1227.5 KB)
        "movabs $81920, %%r10;" // reset-counter for RAM-buffer with 20 cache lines accessed per loop (102400.0 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_snb_corei_avx_1t:"
        /****************************************************************************************************
         decode 0                            decode 1                            decode 2                            decode 3 */
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rdx);          psllw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm7, %%ymm7;                                       psllw %%mm4, %%mm1;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rdx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rdx);          psllw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm2, %%ymm2;                                       psrlw %%mm1, %%mm4;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rdx);          psllw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm8, %%ymm8;                                       psrlw %%mm5, %%mm2;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rdx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm3, %%ymm3;                                       psllw %%mm2, %%mm5;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rdx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psllw %%mm1, %%mm4;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm4, %%mm1;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm2, %%mm5;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psrlw %%mm5, %%mm2;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rdx);          psllw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm2, %%ymm2;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rdx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm7, %%ymm7;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rdx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm3, %%ymm3;                                       psrlw %%mm4, %%mm1;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rdx);          psllw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rdx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm8, %%ymm8;                                       psllw %%mm1, %%mm4;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rdx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psllw %%mm5, %%mm2;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rdx);          psllw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm2, %%mm5;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rdx);          psllw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rdx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psrlw %%mm1, %%mm4;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rdx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        //reset RAM counter
        "sub $1, %%r10;"
        "jnz _work_no_ram_reset_snb_corei_avx_1t;"
        "movabs $81920, %%r10;"
        "mov %%rax, %%rdi;"
        "add $1572864, %%rdi;"
        "_work_no_ram_reset_snb_corei_avx_1t:"
        //reset L2-Cache counter
        "sub $1, %%r8;"
        "jnz _work_no_L2_reset_snb_corei_avx_1t;"
        "movabs $32, %%r8;"
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;"
        "_work_no_L2_reset_snb_corei_avx_1t:"
        //reset L3-Cache counter
        "sub $1, %%r9;"
        "jnz _work_no_L3_reset_snb_corei_avx_1t;"
        "movabs $491, %%r9;"
        "mov %%rax, %%rdx;"
        "add $262144, %%rdx;"
        "_work_no_L3_reset_snb_corei_avx_1t:"
        "inc %%r14;" // increment iteration counter
        "mov %%rax, %%rbx;"
        "testq $1, (%%r13);"
        "jnz _work_loop_snb_corei_avx_1t;"
        "movq %%r14, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
    return EXIT_SUCCESS;
}
/**
 * assembler implementation of processor and memory stress test
 * ISA: AVX
 * optimized for Sandy Bridge - 2 thread(s) per core
 */
int asm_work_snb_corei_avx_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_corei_avx_2t(threaddata_t* threaddata)
{
    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
        /* input: 
         *   - threaddata->addrMem    -> rax
         *   - threaddata->addrHigh   -> rbx
         *   - threaddata->iterations -> rcx
         * output: 
         *   - rax -> threaddata->iterations
         * register usage:
         *   - rax:           stores original pointer to buffer, used to periodically reset other pointers
         *   - rbx:           pointer to L1 buffer
         *   - rcx:           pointer to L2 buffer
         *   - rdx:           pointer to L3 buffer
         *   - rdi:           pointer to RAM buffer
         *   - r8:            counter for L2-pointer reset
         *   - r9:            counter for L3-pointer reset
         *   - r10:           counter for RAM-pointer reset
         *   - r11:           register for temporary results
         *   - r12:           stores cacheline width as increment for buffer addresses
         *   - r13:           stores address of shared variable that controls load level
         *   - r14:           stores iteration counter
         *   - mm*,xmm*,ymm*: data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r13;" // store address of shared variable that controls load level in r13
        "mov %%rcx, %%r14;" // store iteration counter in r14
        "mov $64, %%r12;" // increment after each cache/memory access
        //Initialize AVX-Registers for Addition
        "vmovapd 0(%%rax), %%ymm0;"
        "vmovapd 32(%%rax), %%ymm1;"
        "vmovapd 64(%%rax), %%ymm2;"
        "vmovapd 96(%%rax), %%ymm3;"
        "vmovapd 128(%%rax), %%ymm4;"
        "vmovapd 160(%%rax), %%ymm5;"
        "vmovapd 192(%%rax), %%ymm6;"
        "vmovapd 224(%%rax), %%ymm7;"
        "vmovapd 256(%%rax), %%ymm8;"
        "vmovapd 288(%%rax), %%ymm9;"
        //Initialize MMX-Registers for shift operations
        "movabs $0x5555555555555555, %%r11;"
        "movq %%r11, %%mm0;"
        "movq %%mm0, %%mm1;"
        "movq %%mm0, %%mm2;"
        "movq %%mm0, %%mm3;"
        "movq %%mm0, %%mm4;"
        "movq %%mm0, %%mm5;"
        //Initialize AVX-Registers for Transfer-Operations
        "movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
        "pinsrq $0, %%r11, %%xmm10;"
        "pinsrq $1, %%r11, %%xmm10;"
        "vinsertf128 $1, %%xmm10, %%ymm10, %%ymm10;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm11;"
        "pinsrq $1, %%r11, %%xmm11;"
        "vinsertf128 $1, %%xmm11, %%ymm11, %%ymm11;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm12;"
        "pinsrq $1, %%r11, %%xmm12;"
        "vinsertf128 $1, %%xmm12, %%ymm12, %%ymm12;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm13;"
        "pinsrq $1, %%r11, %%xmm13;"
        "vinsertf128 $1, %%xmm13, %%ymm13, %%ymm13;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm14;"
        "pinsrq $1, %%r11, %%xmm14;"
        "vinsertf128 $1, %%xmm14, %%ymm14, %%ymm14;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm15;"
        "pinsrq $1, %%r11, %%xmm15;"
        "vinsertf128 $1, %%xmm15, %%ymm15, %%ymm15;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $16384, %%rcx;" // address for L2-buffer
        "mov %%rax, %%rdx;"
        "add $131072, %%rdx;" // address for L3-buffer
        "mov %%rax, %%rdi;"
        "add $786432, %%rdi;" // address for RAM-buffer
        "movabs $32, %%r8;" // reset-counter for L2-buffer with 50 cache lines accessed per loop (100.0 KB)
        "movabs $491, %%r9;" // reset-counter for L3-buffer with 20 cache lines accessed per loop (613.75 KB)
        "movabs $81920, %%r10;" // reset-counter for RAM-buffer with 10 cache lines accessed per loop (51200.0 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_snb_corei_avx_2t:"
        /****************************************************************************************************
         decode 0                            decode 1                            decode 2                            decode 3 */
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rdx);          psllw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm7, %%ymm7;                                       psllw %%mm4, %%mm1;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rdx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rdx);          psllw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm2, %%ymm2;                                       psrlw %%mm1, %%mm4;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rdx);          psllw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm8, %%ymm8;                                       psrlw %%mm5, %%mm2;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rdx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm3, %%ymm3;                                       psllw %%mm2, %%mm5;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rdx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm2, %%mm5;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psllw %%mm1, %%mm4;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm4, %%mm1;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm1, %%mm4;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm2, %%mm5;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm0, %%mm3;                 vmovdqa %%ymm11, %%ymm10;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm3, %%mm0;                 vmovdqa %%ymm14, %%ymm13;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm5, %%mm2;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        //reset RAM counter
        "sub $1, %%r10;"
        "jnz _work_no_ram_reset_snb_corei_avx_2t;"
        "movabs $81920, %%r10;"
        "mov %%rax, %%rdi;"
        "add $786432, %%rdi;"
        "_work_no_ram_reset_snb_corei_avx_2t:"
        //reset L2-Cache counter
        "sub $1, %%r8;"
        "jnz _work_no_L2_reset_snb_corei_avx_2t;"
        "movabs $32, %%r8;"
        "mov %%rax, %%rcx;"
        "add $16384, %%rcx;"
        "_work_no_L2_reset_snb_corei_avx_2t:"
        //reset L3-Cache counter
        "sub $1, %%r9;"
        "jnz _work_no_L3_reset_snb_corei_avx_2t;"
        "movabs $491, %%r9;"
        "mov %%rax, %%rdx;"
        "add $131072, %%rdx;"
        "_work_no_L3_reset_snb_corei_avx_2t:"
        "inc %%r14;" // increment iteration counter
        "mov %%rax, %%rbx;"
        "testq $1, (%%r13);"
        "jnz _work_loop_snb_corei_avx_2t;"
        "movq %%r14, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
    return EXIT_SUCCESS;
}
/**
 * assembler implementation of processor and memory stress test
 * ISA: AVX
 * optimized for Sandy Bridge-EP - 1 thread(s) per core
 */
int asm_work_snb_xeonep_avx_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_xeonep_avx_1t(threaddata_t* threaddata)
{
    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
        /* input: 
         *   - threaddata->addrMem    -> rax
         *   - threaddata->addrHigh   -> rbx
         *   - threaddata->iterations -> rcx
         * output: 
         *   - rax -> threaddata->iterations
         * register usage:
         *   - rax:           stores original pointer to buffer, used to periodically reset other pointers
         *   - rbx:           pointer to L1 buffer
         *   - rcx:           pointer to L2 buffer
         *   - rdx:           pointer to L3 buffer
         *   - rdi:           pointer to RAM buffer
         *   - r8:            counter for L2-pointer reset
         *   - r9:            counter for L3-pointer reset
         *   - r10:           counter for RAM-pointer reset
         *   - r11:           register for temporary results
         *   - r12:           stores cacheline width as increment for buffer addresses
         *   - r13:           stores address of shared variable that controls load level
         *   - r14:           stores iteration counter
         *   - mm*,xmm*,ymm*: data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r13;" // store address of shared variable that controls load level in r13
        "mov %%rcx, %%r14;" // store iteration counter in r14
        "mov $64, %%r12;" // increment after each cache/memory access
        //Initialize AVX-Registers for Addition
        "vmovapd 0(%%rax), %%ymm0;"
        "vmovapd 32(%%rax), %%ymm1;"
        "vmovapd 64(%%rax), %%ymm2;"
        "vmovapd 96(%%rax), %%ymm3;"
        "vmovapd 128(%%rax), %%ymm4;"
        "vmovapd 160(%%rax), %%ymm5;"
        "vmovapd 192(%%rax), %%ymm6;"
        "vmovapd 224(%%rax), %%ymm7;"
        "vmovapd 256(%%rax), %%ymm8;"
        "vmovapd 288(%%rax), %%ymm9;"
        //Initialize MMX-Registers for shift operations
        "movabs $0x5555555555555555, %%r11;"
        "movq %%r11, %%mm0;"
        "movq %%mm0, %%mm1;"
        "movq %%mm0, %%mm2;"
        "movq %%mm0, %%mm3;"
        "movq %%mm0, %%mm4;"
        "movq %%mm0, %%mm5;"
        //Initialize AVX-Registers for Transfer-Operations
        "movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
        "pinsrq $0, %%r11, %%xmm10;"
        "pinsrq $1, %%r11, %%xmm10;"
        "vinsertf128 $1, %%xmm10, %%ymm10, %%ymm10;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm11;"
        "pinsrq $1, %%r11, %%xmm11;"
        "vinsertf128 $1, %%xmm11, %%ymm11, %%ymm11;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm12;"
        "pinsrq $1, %%r11, %%xmm12;"
        "vinsertf128 $1, %%xmm12, %%ymm12, %%ymm12;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm13;"
        "pinsrq $1, %%r11, %%xmm13;"
        "vinsertf128 $1, %%xmm13, %%ymm13, %%ymm13;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm14;"
        "pinsrq $1, %%r11, %%xmm14;"
        "vinsertf128 $1, %%xmm14, %%ymm14, %%ymm14;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm15;"
        "pinsrq $1, %%r11, %%xmm15;"
        "vinsertf128 $1, %%xmm15, %%ymm15, %%ymm15;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;" // address for L2-buffer
        "mov %%rax, %%rdx;"
        "add $262144, %%rdx;" // address for L3-buffer
        "mov %%rax, %%rdi;"
        "add $2621440, %%rdi;" // address for RAM-buffer
        "movabs $29, %%r8;" // reset-counter for L2-buffer with 110 cache lines accessed per loop (199.38 KB)
        "movabs $1489, %%r9;" // reset-counter for L3-buffer with 22 cache lines accessed per loop (2047.38 KB)
        "movabs $49648, %%r10;" // reset-counter for RAM-buffer with 33 cache lines accessed per loop (102399.0 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_snb_xeonep_avx_1t:"
        /****************************************************************************************************
         decode 0                            decode 1                            decode 2                            decode 3 */
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        //reset RAM counter
        "sub $1, %%r10;"
        "jnz _work_no_ram_reset_snb_xeonep_avx_1t;"
        "movabs $49648, %%r10;"
        "mov %%rax, %%rdi;"
        "add $2621440, %%rdi;"
        "_work_no_ram_reset_snb_xeonep_avx_1t:"
        //reset L2-Cache counter
        "sub $1, %%r8;"
        "jnz _work_no_L2_reset_snb_xeonep_avx_1t;"
        "movabs $29, %%r8;"
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;"
        "_work_no_L2_reset_snb_xeonep_avx_1t:"
        //reset L3-Cache counter
        "sub $1, %%r9;"
        "jnz _work_no_L3_reset_snb_xeonep_avx_1t;"
        "movabs $1489, %%r9;"
        "mov %%rax, %%rdx;"
        "add $262144, %%rdx;"
        "_work_no_L3_reset_snb_xeonep_avx_1t:"
        "inc %%r14;" // increment iteration counter
        "mov %%rax, %%rbx;"
        "testq $1, (%%r13);"
        "jnz _work_loop_snb_xeonep_avx_1t;"
        "movq %%r14, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
    return EXIT_SUCCESS;
}
/**
 * assembler implementation of processor and memory stress test
 * ISA: AVX
 * optimized for Sandy Bridge-EP - 2 thread(s) per core
 */
int asm_work_snb_xeonep_avx_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_snb_xeonep_avx_2t(threaddata_t* threaddata)
{
    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
        /* input: 
         *   - threaddata->addrMem    -> rax
         *   - threaddata->addrHigh   -> rbx
         *   - threaddata->iterations -> rcx
         * output: 
         *   - rax -> threaddata->iterations
         * register usage:
         *   - rax:           stores original pointer to buffer, used to periodically reset other pointers
         *   - rbx:           pointer to L1 buffer
         *   - rcx:           pointer to L2 buffer
         *   - rdx:           pointer to L3 buffer
         *   - rdi:           pointer to RAM buffer
         *   - r8:            counter for L2-pointer reset
         *   - r9:            counter for L3-pointer reset
         *   - r10:           counter for RAM-pointer reset
         *   - r11:           register for temporary results
         *   - r12:           stores cacheline width as increment for buffer addresses
         *   - r13:           stores address of shared variable that controls load level
         *   - r14:           stores iteration counter
         *   - mm*,xmm*,ymm*: data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r13;" // store address of shared variable that controls load level in r13
        "mov %%rcx, %%r14;" // store iteration counter in r14
        "mov $64, %%r12;" // increment after each cache/memory access
        //Initialize AVX-Registers for Addition
        "vmovapd 0(%%rax), %%ymm0;"
        "vmovapd 32(%%rax), %%ymm1;"
        "vmovapd 64(%%rax), %%ymm2;"
        "vmovapd 96(%%rax), %%ymm3;"
        "vmovapd 128(%%rax), %%ymm4;"
        "vmovapd 160(%%rax), %%ymm5;"
        "vmovapd 192(%%rax), %%ymm6;"
        "vmovapd 224(%%rax), %%ymm7;"
        "vmovapd 256(%%rax), %%ymm8;"
        "vmovapd 288(%%rax), %%ymm9;"
        //Initialize MMX-Registers for shift operations
        "movabs $0x5555555555555555, %%r11;"
        "movq %%r11, %%mm0;"
        "movq %%mm0, %%mm1;"
        "movq %%mm0, %%mm2;"
        "movq %%mm0, %%mm3;"
        "movq %%mm0, %%mm4;"
        "movq %%mm0, %%mm5;"
        //Initialize AVX-Registers for Transfer-Operations
        "movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
        "pinsrq $0, %%r11, %%xmm10;"
        "pinsrq $1, %%r11, %%xmm10;"
        "vinsertf128 $1, %%xmm10, %%ymm10, %%ymm10;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm11;"
        "pinsrq $1, %%r11, %%xmm11;"
        "vinsertf128 $1, %%xmm11, %%ymm11, %%ymm11;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm12;"
        "pinsrq $1, %%r11, %%xmm12;"
        "vinsertf128 $1, %%xmm12, %%ymm12, %%ymm12;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm13;"
        "pinsrq $1, %%r11, %%xmm13;"
        "vinsertf128 $1, %%xmm13, %%ymm13, %%ymm13;"
        "shr $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm14;"
        "pinsrq $1, %%r11, %%xmm14;"
        "vinsertf128 $1, %%xmm14, %%ymm14, %%ymm14;"
        "shl $4, %%r11;"
        "pinsrq $0, %%r11, %%xmm15;"
        "pinsrq $1, %%r11, %%xmm15;"
        "vinsertf128 $1, %%xmm15, %%ymm15, %%ymm15;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $16384, %%rcx;" // address for L2-buffer
        "mov %%rax, %%rdx;"
        "add $131072, %%rdx;" // address for L3-buffer
        "mov %%rax, %%rdi;"
        "add $1310720, %%rdi;" // address for RAM-buffer
        "movabs $32, %%r8;" // reset-counter for L2-buffer with 50 cache lines accessed per loop (100.0 KB)
        "movabs $1638, %%r9;" // reset-counter for L3-buffer with 10 cache lines accessed per loop (1023.75 KB)
        "movabs $54613, %%r10;" // reset-counter for RAM-buffer with 15 cache lines accessed per loop (51199.69 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_snb_xeonep_avx_2t:"
        /****************************************************************************************************
         decode 0                            decode 1                            decode 2                            decode 3 */
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm6, %%ymm6;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rdx);          psllw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm0, %%ymm0;                                       psllw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rdx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm4, %%ymm4;                                       psrlw %%mm3, %%mm0;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm1, %%ymm1;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rdx);          psllw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm5, %%mm2;                 mov %%rax, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm5, %%ymm5;                                       psrlw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 96(%%rcx);          psllw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 96(%%rdx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rdx;                  " // L3 load, L3 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm6, %%ymm0, %%ymm7;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm5, %%ymm0, %%ymm6;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 96(%%rcx);          psllw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm0, %%ymm0, %%ymm1;                                          psllw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm4, %%ymm0, %%ymm5;                                          psrlw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rdi), %%ymm9, %%ymm9;                                       psllw %%mm0, %%mm3;                 add %%r12, %%rdi;                  " // RAM load
        "vaddpd %%ymm9, %%ymm0, %%ymm0;                                          psllw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm3, %%ymm0, %%ymm4;                                          psllw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psrlw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm8, %%ymm8;   vmovapd %%xmm8, 96(%%rcx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm8, %%ymm0, %%ymm9;                                          psrlw %%mm4, %%mm1;                 vmovdqa %%ymm15, %%ymm14;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm2, %%ymm2;   vmovapd %%xmm2, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm2, %%ymm0, %%ymm3;                                          psllw %%mm2, %%mm5;                 vmovdqa %%ymm13, %%ymm12;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm6, %%ymm6;   vmovapd %%xmm6, 64(%%rbx);          psllw %%mm5, %%mm2;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 64(%%rcx), %%ymm7, %%ymm7;   vmovapd %%xmm7, 96(%%rcx);          psrlw %%mm0, %%mm3;                 add %%r12, %%rcx;                  " // L2 load, L2 store
        "vaddpd %%ymm7, %%ymm0, %%ymm8;                                          psrlw %%mm1, %%mm4;                 vmovdqa %%ymm12, %%ymm11;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm9, %%ymm9;   vmovapd %%xmm9, 64(%%rbx);          psrlw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm0, %%ymm0;   vmovapd %%xmm0, 64(%%rbx);          psrlw %%mm3, %%mm0;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm1, %%ymm1;   vmovapd %%xmm1, 64(%%rbx);          psrlw %%mm4, %%mm1;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd %%ymm1, %%ymm0, %%ymm2;                                          psrlw %%mm5, %%mm2;                 vmovdqa %%ymm10, %%ymm15;          " // REG ops only
        "vaddpd 32(%%rbx), %%ymm3, %%ymm3;   vmovapd %%xmm3, 64(%%rbx);          psllw %%mm0, %%mm3;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm4, %%ymm4;   vmovapd %%xmm4, 64(%%rbx);          psllw %%mm1, %%mm4;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        "vaddpd 32(%%rbx), %%ymm5, %%ymm5;   vmovapd %%xmm5, 64(%%rbx);          psllw %%mm2, %%mm5;                 add %%r12, %%rbx;                  " // L1 load, L1 store
        //reset RAM counter
        "sub $1, %%r10;"
        "jnz _work_no_ram_reset_snb_xeonep_avx_2t;"
        "movabs $54613, %%r10;"
        "mov %%rax, %%rdi;"
        "add $1310720, %%rdi;"
        "_work_no_ram_reset_snb_xeonep_avx_2t:"
        //reset L2-Cache counter
        "sub $1, %%r8;"
        "jnz _work_no_L2_reset_snb_xeonep_avx_2t;"
        "movabs $32, %%r8;"
        "mov %%rax, %%rcx;"
        "add $16384, %%rcx;"
        "_work_no_L2_reset_snb_xeonep_avx_2t:"
        //reset L3-Cache counter
        "sub $1, %%r9;"
        "jnz _work_no_L3_reset_snb_xeonep_avx_2t;"
        "movabs $1638, %%r9;"
        "mov %%rax, %%rdx;"
        "add $131072, %%rdx;"
        "_work_no_L3_reset_snb_xeonep_avx_2t:"
        "inc %%r14;" // increment iteration counter
        "mov %%rax, %%rbx;"
        "testq $1, (%%r13);"
        "jnz _work_loop_snb_xeonep_avx_2t;"
        "movq %%r14, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
    return EXIT_SUCCESS;
}

