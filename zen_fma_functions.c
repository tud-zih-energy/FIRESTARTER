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
 * ISA: ZEN_FMA
 * optimized for Naples - 1 thread(s) per core
 */
int asm_work_zen_epyc_zen_fma_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_zen_epyc_zen_fma_1t(threaddata_t* threaddata)
{
    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
        /* input: 
         *   - threaddata->addrMem    -> rax
         *   - threaddata->addrHigh   -> rbx
         *   - threaddata->iterations -> rcx
         * output: 
         *   - rax -> threaddata->iterations
         * register usage:
         *   - rax:         stores original pointer to buffer, used to periodically reset other pointers
         *   - rbx:         pointer to L1 buffer
         *   - rcx:         pointer to L2 buffer
         *   - r8:          pointer to L3 buffer
         *   - r9:          pointer to RAM buffer
         *   - r10:         counter for L2-pointer reset
         *   - r11:         counter for L3-pointer reset
         *   - r12:         counter for RAM-pointer reset
         *   - r13:         register for temporary results
         *   - r14:         stores cacheline width as increment for buffer addresses
         *   - r15:         stores address of shared variable that controls load level
         *   - mm0:         stores iteration counter
         *   - rdi,rsi,rdx: registers for shift operations
         *   - ymm*,ymm*:   data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r15;" // store address of shared variable that controls load level in r15
        "movq %%rcx, %%mm0;" // store iteration counter in mm0
        "mov $64, %%r14;" // increment after each cache/memory access
        //Initialize registers for shift operations
        "mov $0xAAAAAAAAAAAAAAAA, %%rdi;"
        "mov $0xAAAAAAAAAAAAAAAA, %%rsi;"
        "mov $0xAAAAAAAAAAAAAAAA, %%rdx;"
        //Initialize AVX-Registers for FMA Operations
        "vmovapd (%%rax), %%ymm0;"
        "vmovapd 32(%%rax), %%ymm1;"
        "vmovapd 64(%%rax), %%ymm2;"
        "vmovapd 96(%%rax), %%ymm3;"
        "vmovapd 128(%%rax), %%ymm4;"
        "vmovapd 160(%%rax), %%ymm5;"
        "vmovapd 192(%%rax), %%ymm6;"
        "vmovapd 224(%%rax), %%ymm7;"
        "vmovapd 256(%%rax), %%ymm8;"
        "vmovapd 288(%%rax), %%ymm9;"
        "vmovapd 320(%%rax), %%ymm10;"
        "vmovapd 352(%%rax), %%ymm11;"
        "mov $1, %%r13;"
        "movd %%xmm14, %%r13;"
        "movd %%xmm13, %%rdi;"
        "vbroadcastss %%xmm13, %%xmm13;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $65536, %%rcx;" // address for L2-buffer
        "mov %%rax, %%r8;"
        "add $524288, %%r8;" // address for L3-buffer
        "mov %%rax, %%r9;"
        "add $2097152, %%r9;" // address for RAM-buffer
        "movabs $16, %%r10;" // reset-counter for L2-buffer with 405 cache lines accessed per loop (405.0 KB)
        "movabs $158, %%r11;" // reset-counter for L3-buffer with 165 cache lines accessed per loop (1629.38 KB)
        "movabs $40960, %%r12;" // reset-counter for RAM-buffer with 40 cache lines accessed per loop (102400.0 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_zen_epyc_zen_fma_1t:"
        /****************************************************************************************************
         decode 0                                 decode 1                                 decode 2             decode 3 */
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm10, %%ymm0, %%ymm4;     xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm7, %%ymm0, %%ymm1;      xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "movq %%mm0, %%r13;" // restore iteration counter
        //reset RAM counter
        "sub $1, %%r12;"
        "jnz _work_no_ram_reset_zen_epyc_zen_fma_1t;"
        "movabs $40960, %%r12;"
        "mov %%rax, %%r9;"
        "add $2097152, %%r9;"
        "_work_no_ram_reset_zen_epyc_zen_fma_1t:"
        "inc %%r13;" // increment iteration counter
        //reset L2-Cache counter
        "sub $1, %%r10;"
        "jnz _work_no_L2_reset_zen_epyc_zen_fma_1t;"
        "movabs $16, %%r10;"
        "mov %%rax, %%rcx;"
        "add $65536, %%rcx;"
        "_work_no_L2_reset_zen_epyc_zen_fma_1t:"
        "movq %%r13, %%mm0;" // store iteration counter
        //reset L3-Cache counter
        "sub $1, %%r11;"
        "jnz _work_no_L3_reset_zen_epyc_zen_fma_1t;"
        "movabs $158, %%r11;"
        "mov %%rax, %%r8;"
        "add $524288, %%r8;"
        "_work_no_L3_reset_zen_epyc_zen_fma_1t:"
        "mov %%rax, %%rbx;"
        "testq $1, (%%r15);"
        "jnz _work_loop_zen_epyc_zen_fma_1t;"
        "movq %%mm0, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%rdi", "%rsi", "%rdx", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15"
        );
    return EXIT_SUCCESS;
}
/**
 * assembler implementation of processor and memory stress test
 * ISA: ZEN_FMA
 * optimized for Naples - 2 thread(s) per core
 */
int asm_work_zen_epyc_zen_fma_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_zen_epyc_zen_fma_2t(threaddata_t* threaddata)
{
    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
        /* input: 
         *   - threaddata->addrMem    -> rax
         *   - threaddata->addrHigh   -> rbx
         *   - threaddata->iterations -> rcx
         * output: 
         *   - rax -> threaddata->iterations
         * register usage:
         *   - rax:         stores original pointer to buffer, used to periodically reset other pointers
         *   - rbx:         pointer to L1 buffer
         *   - rcx:         pointer to L2 buffer
         *   - r8:          pointer to L3 buffer
         *   - r9:          pointer to RAM buffer
         *   - r10:         counter for L2-pointer reset
         *   - r11:         counter for L3-pointer reset
         *   - r12:         counter for RAM-pointer reset
         *   - r13:         register for temporary results
         *   - r14:         stores cacheline width as increment for buffer addresses
         *   - r15:         stores address of shared variable that controls load level
         *   - mm0:         stores iteration counter
         *   - rdi,rsi,rdx: registers for shift operations
         *   - ymm*,ymm*:   data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r15;" // store address of shared variable that controls load level in r15
        "movq %%rcx, %%mm0;" // store iteration counter in mm0
        "mov $64, %%r14;" // increment after each cache/memory access
        //Initialize registers for shift operations
        "mov $0xAAAAAAAAAAAAAAAA, %%rdi;"
        "mov $0xAAAAAAAAAAAAAAAA, %%rsi;"
        "mov $0xAAAAAAAAAAAAAAAA, %%rdx;"
        //Initialize AVX-Registers for FMA Operations
        "vmovapd (%%rax), %%ymm0;"
        "vmovapd 32(%%rax), %%ymm1;"
        "vmovapd 64(%%rax), %%ymm2;"
        "vmovapd 96(%%rax), %%ymm3;"
        "vmovapd 128(%%rax), %%ymm4;"
        "vmovapd 160(%%rax), %%ymm5;"
        "vmovapd 192(%%rax), %%ymm6;"
        "vmovapd 224(%%rax), %%ymm7;"
        "vmovapd 256(%%rax), %%ymm8;"
        "vmovapd 288(%%rax), %%ymm9;"
        "vmovapd 320(%%rax), %%ymm10;"
        "vmovapd 352(%%rax), %%ymm11;"
        "mov $1, %%r13;"
        "movd %%xmm14, %%r13;"
        "movd %%xmm13, %%rdi;"
        "vbroadcastss %%xmm13, %%xmm13;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;" // address for L2-buffer
        "mov %%rax, %%r8;"
        "add $262144, %%r8;" // address for L3-buffer
        "mov %%rax, %%r9;"
        "add $1048576, %%r9;" // address for RAM-buffer
        "movabs $20, %%r10;" // reset-counter for L2-buffer with 162 cache lines accessed per loop (202.5 KB)
        "movabs $198, %%r11;" // reset-counter for L3-buffer with 66 cache lines accessed per loop (816.75 KB)
        "movabs $51200, %%r12;" // reset-counter for RAM-buffer with 16 cache lines accessed per loop (51200.0 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_zen_epyc_zen_fma_2t:"
        /****************************************************************************************************
         decode 0                                 decode 1                                 decode 2             decode 3 */
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm6;    xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm3;   vmovapd %%ymm3, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm3;    xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm6;   xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm8, %%ymm0, %%ymm2;      xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rsi;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm3;   xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm11, %%ymm0, %%ymm5;     xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rsi;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm6;   vmovapd %%ymm6, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%r9), %%ymm0, %%ymm15;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r9;    " // RAM load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm1;    xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm2;    xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm4;   vmovapd %%ymm4, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm5;   xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm4;    xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm5;   vmovapd %%ymm5, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm1;   xor %%rdx, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm2;   vmovapd %%ymm2, 64(%%rbx);               vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd %%ymm9, %%ymm0, %%ymm3;      xor %%rsi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; shl $1, %%rdx;      " // REG ops only
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm4;   xor %%rdx, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd 64(%%r8), %%ymm0, %%ymm5;    xor %%rdi, %%r13;                        vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%r8;    " // L3 load
        "vfmadd231pd %%ymm12, %%ymm0, %%ymm6;     xor %%rsi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; shr $1, %%rdx;      " // REG ops only
        "vfmadd231pd 32(%%rbx), %%ymm0, %%ymm1;   vmovapd %%ymm1, 64(%%rbx);               vpsrlq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rbx;   " // L1 load, L1 store
        "vfmadd231pd 64(%%rcx), %%ymm0, %%ymm2;   xor %%rdi, %%r13;                        vpsllq %%xmm13, %%xmm13, %%xmm14; add %%r14, %%rcx;   " // L2 load
        "movq %%mm0, %%r13;" // restore iteration counter
        //reset RAM counter
        "sub $1, %%r12;"
        "jnz _work_no_ram_reset_zen_epyc_zen_fma_2t;"
        "movabs $51200, %%r12;"
        "mov %%rax, %%r9;"
        "add $1048576, %%r9;"
        "_work_no_ram_reset_zen_epyc_zen_fma_2t:"
        "inc %%r13;" // increment iteration counter
        //reset L2-Cache counter
        "sub $1, %%r10;"
        "jnz _work_no_L2_reset_zen_epyc_zen_fma_2t;"
        "movabs $20, %%r10;"
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;"
        "_work_no_L2_reset_zen_epyc_zen_fma_2t:"
        "movq %%r13, %%mm0;" // store iteration counter
        //reset L3-Cache counter
        "sub $1, %%r11;"
        "jnz _work_no_L3_reset_zen_epyc_zen_fma_2t;"
        "movabs $198, %%r11;"
        "mov %%rax, %%r8;"
        "add $262144, %%r8;"
        "_work_no_L3_reset_zen_epyc_zen_fma_2t:"
        "mov %%rax, %%rbx;"
        "testq $1, (%%r15);"
        "jnz _work_loop_zen_epyc_zen_fma_2t;"
        "movq %%mm0, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%rdi", "%rsi", "%rdx", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15"
        );
    return EXIT_SUCCESS;
}

