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
 * ISA: AVX512
 * optimized for Knights_Landing - 4 thread(s) per core
 */
int asm_work_knl_xeonphi_avx512_4t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_knl_xeonphi_avx512_4t(threaddata_t* threaddata)
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
         *   - xmm*,zmm*:   data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r15;" // store address of shared variable that controls load level in r15
        "movq %%rcx, %%mm0;" // store iteration counter in mm0
        "mov $64, %%r14;" // increment after each cache/memory access
        //Initialize registers for shift operations
        "mov $0xAAAAAAAA, %%edi;"
        "mov $0xAAAAAAAA, %%esi;"
        "mov $0xAAAAAAAA, %%edx;"
        //Initialize AVX512-Registers for FMA Operations
        "vmovapd (%%rax), %%zmm0;"
        "vmovapd (%%rax), %%zmm1;"
        "vmovapd 384(%%rax), %%zmm2;"
        "vmovapd 448(%%rax), %%zmm3;"
        "vmovapd 512(%%rax), %%zmm4;"
        "vmovapd 576(%%rax), %%zmm5;"
        "vmovapd 640(%%rax), %%zmm6;"
        "vmovapd 704(%%rax), %%zmm7;"
        "vmovapd 768(%%rax), %%zmm8;"
        "vmovapd 832(%%rax), %%zmm9;"
        "vmovapd 896(%%rax), %%zmm10;"
        "vmovapd 960(%%rax), %%zmm11;"
        "vmovapd 1024(%%rax), %%zmm12;"
        "vmovapd 1088(%%rax), %%zmm13;"
        "vmovapd 1152(%%rax), %%zmm14;"
        "vmovapd 1216(%%rax), %%zmm15;"
        "vmovapd 1280(%%rax), %%zmm16;"
        "vmovapd 1344(%%rax), %%zmm17;"
        "vmovapd 1408(%%rax), %%zmm18;"
        "vmovapd 1472(%%rax), %%zmm19;"
        "vmovapd 1536(%%rax), %%zmm20;"
        "vmovapd 1600(%%rax), %%zmm21;"
        "vmovapd 1664(%%rax), %%zmm22;"
        "vmovapd 1728(%%rax), %%zmm23;"
        "vmovapd 1792(%%rax), %%zmm24;"
        "vmovapd 1856(%%rax), %%zmm25;"
        "vmovapd 1920(%%rax), %%zmm26;"
        "vmovapd 1984(%%rax), %%zmm27;"
        "vmovapd 2048(%%rax), %%zmm28;"
        "vmovapd 2112(%%rax), %%zmm29;"
        "vmovapd 2176(%%rax), %%zmm30;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $8192, %%rcx;" // address for L2-buffer
        "mov %%rax, %%r8;"
        "add $131072, %%r8;" // address for L3-buffer
        "mov %%rax, %%r9;"
        "add $59069781, %%r9;" // address for RAM-buffer
        "movabs $34, %%r10;" // reset-counter for L2-buffer with 48 cache lines accessed per loop (102.0 KB)
        "movabs $0, %%r11;" // reset-counter for L3-buffer with 0 cache lines accessed per loop (0.0 KB)
        "movabs $5688, %%r12;" // reset-counter for RAM-buffer with 18 cache lines accessed per loop (6399.0 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_knl_xeonphi_avx512_4t:"
        /****************************************************************************************************
         decode 0                                 decode 1                                 decode 2             decode 3 */
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm3;   prefetcht2 (%%r9);                       shl $1, %%edi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm26;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm9, 64(%%rcx);               vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm15, 64(%%rcx);              vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm20, 64(%%rcx);              vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    shr $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm2;   prefetcht2 (%%r9);                       shr $1, %%edx;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm30;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm8, 64(%%rcx);               vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      shr $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm13, 64(%%rcx);              vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm19, 64(%%rcx);              vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm25;  prefetcht2 (%%r9);                       shr $1, %%esi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm29;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm6, 64(%%rcx);               vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      shr $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm12, 64(%%rcx);              vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    shr $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm16;  prefetcht2 (%%r9);                       shl $1, %%esi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm22, 64(%%rcx);              vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    shl $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm4, 64(%%rcx);               vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      shl $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm28;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm9, 64(%%rcx);               vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm15;  prefetcht2 (%%r9);                       shl $1, %%edi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm21, 64(%%rcx);              vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm2, 64(%%rcx);               vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      shr $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       mov %%rax, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm27;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm8, 64(%%rcx);               vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      shr $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm14;  prefetcht2 (%%r9);                       shr $1, %%edx;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm19, 64(%%rcx);              vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm25, 64(%%rcx);              vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm5;   prefetcht2 (%%r9);                       shl $1, %%edx;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm11, 64(%%rcx);              vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm17, 64(%%rcx);              vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm22, 64(%%rcx);              vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    shl $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm4;   prefetcht2 (%%r9);                       shl $1, %%esi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm10, 64(%%rcx);              vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    shl $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm15, 64(%%rcx);              vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm21, 64(%%rcx);              vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm3;   prefetcht2 (%%r9);                       shl $1, %%edi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm8, 64(%%rcx);               vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      shr $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm14, 64(%%rcx);              vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    shr $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm18;  prefetcht2 (%%r9);                       shr $1, %%edi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm24, 64(%%rcx);              vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    shr $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       mov %%rax, %%rbx;   " // L1 load
        "vmovapd %%zmm6, 64(%%rcx);               vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      shr $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm11, 64(%%rcx);              vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm17;  prefetcht2 (%%r9);                       shl $1, %%edx;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm23, 64(%%rcx);              vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm4, 64(%%rcx);               vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      shl $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm10, 64(%%rcx);              vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    shl $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm16;  prefetcht2 (%%r9);                       shl $1, %%esi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm21, 64(%%rcx);              vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm3, 64(%%rcx);               vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm7;   prefetcht2 (%%r9);                       shr $1, %%esi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm13, 64(%%rcx);              vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm19, 64(%%rcx);              vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm24, 64(%%rcx);              vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    shr $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm29;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm6;   prefetcht2 (%%r9);                       shr $1, %%edi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm12, 64(%%rcx);              vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    shr $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm17, 64(%%rcx);              vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm23, 64(%%rcx);              vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm28;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm5;   prefetcht2 (%%r9);                       shl $1, %%edx;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       mov %%rax, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm10, 64(%%rcx);              vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    shl $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm16, 64(%%rcx);              vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    shl $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm20;  prefetcht2 (%%r9);                       shr $1, %%edx;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm2, 64(%%rcx);               vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      shr $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm27;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm8, 64(%%rcx);               vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      shr $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm13, 64(%%rcx);              vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm19;  prefetcht2 (%%r9);                       shr $1, %%esi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm25, 64(%%rcx);              vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm26;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm6, 64(%%rcx);               vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      shr $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm12, 64(%%rcx);              vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    shr $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm18;  prefetcht2 (%%r9);                       shr $1, %%edi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm23, 64(%%rcx);              vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm30;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vmovapd %%zmm5, 64(%%rcx);               vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "movq %%mm0, %%r13;" // restore iteration counter
        //reset RAM counter
        "sub $1, %%r12;"
        "jnz _work_no_ram_reset_knl_xeonphi_avx512_4t;"
        "movabs $5688, %%r12;"
        "mov %%rax, %%r9;"
        "add $59069781, %%r9;"
        "_work_no_ram_reset_knl_xeonphi_avx512_4t:"
        "inc %%r13;" // increment iteration counter
        //reset L2-Cache counter
        "sub $1, %%r10;"
        "jnz _work_no_L2_reset_knl_xeonphi_avx512_4t;"
        "movabs $34, %%r10;"
        "mov %%rax, %%rcx;"
        "add $8192, %%rcx;"
        "_work_no_L2_reset_knl_xeonphi_avx512_4t:"
        "movq %%r13, %%mm0;" // store iteration counter
        "mov %%rax, %%rbx;"
        "testq $1, (%%r15);"
        "jnz _work_loop_knl_xeonphi_avx512_4t;"
        "movq %%mm0, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%rdi", "%rsi", "%rdx", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
    return EXIT_SUCCESS;
}
/**
 * assembler implementation of processor and memory stress test
 * ISA: AVX512
 * optimized for Skylake-SP - 1 thread(s) per core
 */
int asm_work_skl_xeonep_avx512_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_skl_xeonep_avx512_1t(threaddata_t* threaddata)
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
         *   - xmm*,zmm*:   data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r15;" // store address of shared variable that controls load level in r15
        "movq %%rcx, %%mm0;" // store iteration counter in mm0
        "mov $64, %%r14;" // increment after each cache/memory access
        //Initialize registers for shift operations
        "mov $0xAAAAAAAA, %%edi;"
        "mov $0xAAAAAAAA, %%esi;"
        "mov $0xAAAAAAAA, %%edx;"
        //Initialize AVX512-Registers for FMA Operations
        "vmovapd (%%rax), %%zmm0;"
        "vmovapd (%%rax), %%zmm1;"
        "vmovapd 384(%%rax), %%zmm2;"
        "vmovapd 448(%%rax), %%zmm3;"
        "vmovapd 512(%%rax), %%zmm4;"
        "vmovapd 576(%%rax), %%zmm5;"
        "vmovapd 640(%%rax), %%zmm6;"
        "vmovapd 704(%%rax), %%zmm7;"
        "vmovapd 768(%%rax), %%zmm8;"
        "vmovapd 832(%%rax), %%zmm9;"
        "vmovapd 896(%%rax), %%zmm10;"
        "vmovapd 960(%%rax), %%zmm11;"
        "vmovapd 1024(%%rax), %%zmm12;"
        "vmovapd 1088(%%rax), %%zmm13;"
        "vmovapd 1152(%%rax), %%zmm14;"
        "vmovapd 1216(%%rax), %%zmm15;"
        "vmovapd 1280(%%rax), %%zmm16;"
        "vmovapd 1344(%%rax), %%zmm17;"
        "vmovapd 1408(%%rax), %%zmm18;"
        "vmovapd 1472(%%rax), %%zmm19;"
        "vmovapd 1536(%%rax), %%zmm20;"
        "vmovapd 1600(%%rax), %%zmm21;"
        "vmovapd 1664(%%rax), %%zmm22;"
        "vmovapd 1728(%%rax), %%zmm23;"
        "vmovapd 1792(%%rax), %%zmm24;"
        "vmovapd 1856(%%rax), %%zmm25;"
        "vmovapd 1920(%%rax), %%zmm26;"
        "vmovapd 1984(%%rax), %%zmm27;"
        "vmovapd 2048(%%rax), %%zmm28;"
        "vmovapd 2112(%%rax), %%zmm29;"
        "vmovapd 2176(%%rax), %%zmm30;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;" // address for L2-buffer
        "mov %%rax, %%r8;"
        "add $1048576, %%r8;" // address for L3-buffer
        "mov %%rax, %%r9;"
        "add $1441792, %%r9;" // address for RAM-buffer
        "movabs $59, %%r10;" // reset-counter for L2-buffer with 222 cache lines accessed per loop (818.63 KB)
        "movabs $3003, %%r11;" // reset-counter for L3-buffer with 6 cache lines accessed per loop (1126.13 KB)
        "movabs $1365333, %%r12;" // reset-counter for RAM-buffer with 12 cache lines accessed per loop (1023999.75 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_skl_xeonep_avx512_1t:"
        /****************************************************************************************************
         decode 0                                 decode 1                                 decode 2             decode 3 */
        "vmovapd %%zmm3, 64(%%r9);                vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm26;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm30;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm26;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%rcx);              vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm30;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm29;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm30;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm11;  prefetcht2 (%%r8);                       shl $1, %%edx;       add %%r14, %%r8;    " // L3 prefetch
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm29;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm30;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%rcx);               vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm29;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm28;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm29;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%r8);               vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%r8;    " // L3 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm28;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm27;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm28;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm11;  prefetcht2 (%%r9);                       shl $1, %%edx;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm27;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm28;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%rcx);               vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm27;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm26;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm27;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%r9);               vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm26;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm30;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm26;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%rcx);              vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm30;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm26;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%r9);                vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm30;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm29;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm30;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%r9);               vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm29;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm28;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm29;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%rcx);              vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm28;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm29;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm3;   prefetcht2 (%%r8);                       shl $1, %%edi;       add %%r14, %%r8;    " // L3 prefetch
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm28;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm27;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm28;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%rcx);              vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm27;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm26;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm27;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%r8);               vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%r8;    " // L3 store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm26;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm27;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm3;   prefetcht2 (%%r9);                       shl $1, %%edi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm26;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm30;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm26;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       mov %%rax, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%rcx);              vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm30;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm29;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm30;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%r9);               vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm29;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm30;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%rcx);               vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm29;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm28;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm29;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%r9);               vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm28;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm27;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm28;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%r9);               vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm27;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm28;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%rcx);               vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm27;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm26;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm27;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm19;  prefetcht2 (%%r8);                       shr $1, %%esi;       add %%r14, %%r8;    " // L3 prefetch
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm26;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm30;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm26;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%rcx);              vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm30;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm26;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%r8);                vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%r8;    " // L3 store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm30;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm29;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm30;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm19;  prefetcht2 (%%r9);                       shr $1, %%esi;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm29;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm28;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm29;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%rcx);              vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm28;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm29;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%r9);                vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm28;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm27;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm28;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%rcx);              vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm27;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm26;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm27;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%r9);               vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm26;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm27;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "movq %%mm0, %%r13;" // restore iteration counter
        //reset RAM counter
        "sub $1, %%r12;"
        "jnz _work_no_ram_reset_skl_xeonep_avx512_1t;"
        "movabs $1365333, %%r12;"
        "mov %%rax, %%r9;"
        "add $1441792, %%r9;"
        "_work_no_ram_reset_skl_xeonep_avx512_1t:"
        "inc %%r13;" // increment iteration counter
        //reset L2-Cache counter
        "sub $1, %%r10;"
        "jnz _work_no_L2_reset_skl_xeonep_avx512_1t;"
        "movabs $59, %%r10;"
        "mov %%rax, %%rcx;"
        "add $32768, %%rcx;"
        "_work_no_L2_reset_skl_xeonep_avx512_1t:"
        "movq %%r13, %%mm0;" // store iteration counter
        //reset L3-Cache counter
        "sub $1, %%r11;"
        "jnz _work_no_L3_reset_skl_xeonep_avx512_1t;"
        "movabs $3003, %%r11;"
        "mov %%rax, %%r8;"
        "add $1048576, %%r8;"
        "_work_no_L3_reset_skl_xeonep_avx512_1t:"
        "mov %%rax, %%rbx;"
        "testq $1, (%%r15);"
        "jnz _work_loop_skl_xeonep_avx512_1t;"
        "movq %%mm0, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%rdi", "%rsi", "%rdx", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
    return EXIT_SUCCESS;
}
/**
 * assembler implementation of processor and memory stress test
 * ISA: AVX512
 * optimized for Skylake-SP - 2 thread(s) per core
 */
int asm_work_skl_xeonep_avx512_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_skl_xeonep_avx512_2t(threaddata_t* threaddata)
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
         *   - xmm*,zmm*:   data registers for SIMD instructions
         */
        __asm__ __volatile__(
        "mov %%rax, %%rax;" // store start address of buffer in rax
        "mov %%rbx, %%r15;" // store address of shared variable that controls load level in r15
        "movq %%rcx, %%mm0;" // store iteration counter in mm0
        "mov $64, %%r14;" // increment after each cache/memory access
        //Initialize registers for shift operations
        "mov $0xAAAAAAAA, %%edi;"
        "mov $0xAAAAAAAA, %%esi;"
        "mov $0xAAAAAAAA, %%edx;"
        //Initialize AVX512-Registers for FMA Operations
        "vmovapd (%%rax), %%zmm0;"
        "vmovapd (%%rax), %%zmm1;"
        "vmovapd 384(%%rax), %%zmm2;"
        "vmovapd 448(%%rax), %%zmm3;"
        "vmovapd 512(%%rax), %%zmm4;"
        "vmovapd 576(%%rax), %%zmm5;"
        "vmovapd 640(%%rax), %%zmm6;"
        "vmovapd 704(%%rax), %%zmm7;"
        "vmovapd 768(%%rax), %%zmm8;"
        "vmovapd 832(%%rax), %%zmm9;"
        "vmovapd 896(%%rax), %%zmm10;"
        "vmovapd 960(%%rax), %%zmm11;"
        "vmovapd 1024(%%rax), %%zmm12;"
        "vmovapd 1088(%%rax), %%zmm13;"
        "vmovapd 1152(%%rax), %%zmm14;"
        "vmovapd 1216(%%rax), %%zmm15;"
        "vmovapd 1280(%%rax), %%zmm16;"
        "vmovapd 1344(%%rax), %%zmm17;"
        "vmovapd 1408(%%rax), %%zmm18;"
        "vmovapd 1472(%%rax), %%zmm19;"
        "vmovapd 1536(%%rax), %%zmm20;"
        "vmovapd 1600(%%rax), %%zmm21;"
        "vmovapd 1664(%%rax), %%zmm22;"
        "vmovapd 1728(%%rax), %%zmm23;"
        "vmovapd 1792(%%rax), %%zmm24;"
        "vmovapd 1856(%%rax), %%zmm25;"
        "vmovapd 1920(%%rax), %%zmm26;"
        "vmovapd 1984(%%rax), %%zmm27;"
        "vmovapd 2048(%%rax), %%zmm28;"
        "vmovapd 2112(%%rax), %%zmm29;"
        "vmovapd 2176(%%rax), %%zmm30;"
        "mov %%rax, %%rbx;" // address for L1-buffer
        "mov %%rax, %%rcx;"
        "add $16384, %%rcx;" // address for L2-buffer
        "mov %%rax, %%r8;"
        "add $524288, %%r8;" // address for L3-buffer
        "mov %%rax, %%r9;"
        "add $720896, %%r9;" // address for RAM-buffer
        "movabs $88, %%r10;" // reset-counter for L2-buffer with 74 cache lines accessed per loop (407.0 KB)
        "movabs $4505, %%r11;" // reset-counter for L3-buffer with 2 cache lines accessed per loop (563.13 KB)
        "movabs $2048000, %%r12;" // reset-counter for RAM-buffer with 4 cache lines accessed per loop (512000.0 KB)

        ".align 64;"     /* alignment in bytes */
        "_work_loop_skl_xeonep_avx512_2t:"
        /****************************************************************************************************
         decode 0                                 decode 1                                 decode 2             decode 3 */
        "vmovapd %%zmm3, 64(%%r9);                vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm26;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm30;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm26;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%rcx);              vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm30;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm29;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm30;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm11;  prefetcht2 (%%r8);                       shl $1, %%edx;       add %%r14, %%r8;    " // L3 prefetch
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm29;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm30;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%rcx);               vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm29;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm28;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm29;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%r8);               vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%r8;    " // L3 store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm28;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm27;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm28;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd 64(%%rbx), %%zmm0, %%zmm11;  prefetcht2 (%%r9);                       shl $1, %%edx;       add %%r14, %%r9;    " // RAM prefetch
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm27;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm28;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%rcx);               vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm27;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm26;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm28;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm30;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm26;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm27;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm28;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm30;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm26;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm27;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm19, 64(%%r9);               vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    shr $1, %%esi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vbroadcastsd 64(%%rbx), %%zmm21;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm29;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm30;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm2;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm26;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm27;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vbroadcastsd 64(%%rbx), %%zmm7;          shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm28;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm12;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm30;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm26;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vbroadcastsd 64(%%rbx), %%zmm17;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm27;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm19;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm22;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm29;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vbroadcastsd 64(%%rbx), %%zmm24;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm30;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vbroadcastsd 64(%%rbx), %%zmm3;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm26;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm5;   shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm27;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm8;   shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm28;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vbroadcastsd 64(%%rbx), %%zmm10;         shl $1, %%esi;       mov %%rax, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm11, 64(%%rcx);              vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    shl $1, %%edx;       add %%r14, %%rcx;   " // L2 store
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm29;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vbroadcastsd 64(%%rbx), %%zmm13;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm30;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm15;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm26;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm18;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm27;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vbroadcastsd 64(%%rbx), %%zmm20;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vfmadd231pd %%zmm24, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vbroadcastsd 64(%%rbx), %%zmm23;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm29;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd 64(%%rbx), %%zmm1, %%zmm25;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd %%zmm5, %%zmm1, %%zmm30;     shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm4;   shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm26;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vbroadcastsd 64(%%rbx), %%zmm6;          shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vfmadd231pd %%zmm10, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vbroadcastsd 64(%%rbx), %%zmm9;          shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm28;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm11;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd %%zmm15, %%zmm1, %%zmm29;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm14;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm30;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vbroadcastsd 64(%%rbx), %%zmm16;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vfmadd231pd %%zmm20, %%zmm1, %%zmm26;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vbroadcastsd 64(%%rbx), %%zmm19;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd %%zmm22, %%zmm1, %%zmm27;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm21;  shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd %%zmm25, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm24;  shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vfmadd231pd %%zmm3, %%zmm1, %%zmm29;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vbroadcastsd 64(%%rbx), %%zmm2;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vmovapd %%zmm3, 64(%%r9);                vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      shl $1, %%edi;       add %%r14, %%r9;    " // RAM store
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vfmadd231pd %%zmm6, %%zmm1, %%zmm30;     shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vbroadcastsd 64(%%rbx), %%zmm5;          shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd %%zmm8, %%zmm1, %%zmm26;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm7;   shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd %%zmm11, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm10;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vfmadd231pd %%zmm13, %%zmm1, %%zmm28;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vbroadcastsd 64(%%rbx), %%zmm12;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vfmadd231pd %%zmm16, %%zmm1, %%zmm29;    shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vbroadcastsd 64(%%rbx), %%zmm15;         shl $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd %%zmm18, %%zmm1, %%zmm30;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm17;  shl $1, %%edx;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm20, %%zmm0, %%zmm19;    vfmadd231pd %%zmm21, %%zmm1, %%zmm26;    shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm21, %%zmm0, %%zmm20;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm20;  shr $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm22, %%zmm0, %%zmm21;    vfmadd231pd %%zmm23, %%zmm1, %%zmm27;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm23, %%zmm0, %%zmm22;    vbroadcastsd 64(%%rbx), %%zmm22;         shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm24, %%zmm0, %%zmm23;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm23;  shl $1, %%edx;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm25, %%zmm0, %%zmm24;    vfmadd231pd %%zmm2, %%zmm1, %%zmm28;     shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm2, %%zmm0, %%zmm25;     vbroadcastsd 64(%%rbx), %%zmm25;         shr $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm3, %%zmm0, %%zmm2;      vfmadd231pd %%zmm4, %%zmm1, %%zmm29;     shr $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm4, %%zmm0, %%zmm3;      vfmadd231pd 64(%%rbx), %%zmm1, %%zmm3;   shl $1, %%edi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm5, %%zmm0, %%zmm4;      vbroadcastsd 64(%%rbx), %%zmm4;          shl $1, %%esi;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm6, %%zmm0, %%zmm5;      vfmadd231pd %%zmm7, %%zmm1, %%zmm30;     shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm7, %%zmm0, %%zmm6;      vfmadd231pd 64(%%rcx), %%zmm1, %%zmm6;   shr $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm8, %%zmm0, %%zmm7;      vfmadd231pd %%zmm9, %%zmm1, %%zmm26;     shr $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm9, %%zmm0, %%zmm8;      vbroadcastsd 64(%%rbx), %%zmm8;          shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm10, %%zmm0, %%zmm9;     vfmadd231pd 64(%%rcx), %%zmm1, %%zmm9;   shl $1, %%edi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm11, %%zmm0, %%zmm10;    vfmadd231pd %%zmm12, %%zmm1, %%zmm27;    shl $1, %%esi;       xor %%rdi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm12, %%zmm0, %%zmm11;    vbroadcastsd 64(%%rbx), %%zmm11;         shl $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm13, %%zmm0, %%zmm12;    vfmadd231pd %%zmm14, %%zmm1, %%zmm28;    shr $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm14, %%zmm0, %%zmm13;    vfmadd231pd 64(%%rbx), %%zmm1, %%zmm13;  shr $1, %%esi;       add %%r14, %%rbx;   " // L1 load
        "vfmadd231pd %%zmm15, %%zmm0, %%zmm14;    vbroadcastsd 64(%%rbx), %%zmm14;         shr $1, %%edx;       add %%r14, %%rbx;   " // L1 packed single load
        "vfmadd231pd %%zmm16, %%zmm0, %%zmm15;    vfmadd231pd %%zmm17, %%zmm1, %%zmm29;    shl $1, %%edi;       xor %%rdx, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm17, %%zmm0, %%zmm16;    vfmadd231pd 64(%%rcx), %%zmm1, %%zmm16;  shl $1, %%esi;       add %%r14, %%rcx;   " // L2 load
        "vfmadd231pd %%zmm18, %%zmm0, %%zmm17;    vfmadd231pd %%zmm19, %%zmm1, %%zmm30;    shl $1, %%edx;       xor %%rsi, %%r13;   " // REG ops only
        "vfmadd231pd %%zmm19, %%zmm0, %%zmm18;    vbroadcastsd 64(%%rbx), %%zmm18;         shr $1, %%edi;       add %%r14, %%rbx;   " // L1 packed single load
        "movq %%mm0, %%r13;" // restore iteration counter
        //reset RAM counter
        "sub $1, %%r12;"
        "jnz _work_no_ram_reset_skl_xeonep_avx512_2t;"
        "movabs $2048000, %%r12;"
        "mov %%rax, %%r9;"
        "add $720896, %%r9;"
        "_work_no_ram_reset_skl_xeonep_avx512_2t:"
        "inc %%r13;" // increment iteration counter
        //reset L2-Cache counter
        "sub $1, %%r10;"
        "jnz _work_no_L2_reset_skl_xeonep_avx512_2t;"
        "movabs $88, %%r10;"
        "mov %%rax, %%rcx;"
        "add $16384, %%rcx;"
        "_work_no_L2_reset_skl_xeonep_avx512_2t:"
        "movq %%r13, %%mm0;" // store iteration counter
        //reset L3-Cache counter
        "sub $1, %%r11;"
        "jnz _work_no_L3_reset_skl_xeonep_avx512_2t;"
        "movabs $4505, %%r11;"
        "mov %%rax, %%r8;"
        "add $524288, %%r8;"
        "_work_no_L3_reset_skl_xeonep_avx512_2t:"
        "mov %%rax, %%rbx;"
        "testq $1, (%%r15);"
        "jnz _work_loop_skl_xeonep_avx512_2t;"
        "movq %%mm0, %%rax;" // restore iteration counter
        : "=a" (threaddata->iterations)
        : "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations)
        : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%rdi", "%rsi", "%rdx", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        );
    return EXIT_SUCCESS;
}

