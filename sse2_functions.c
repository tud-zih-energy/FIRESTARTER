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

#include "work.h"




																																																																								int init_nhm_corei_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_corei_sse2_1t(threaddata_t* threaddata)
{
        unsigned long long addrMem = threaddata->addrMem;
	int i;
	for (i = 0; i<13340672; i++) *((double*)(addrMem + 8*i)) = i * 1.654738925401e-15;

        // lines with register operations
        threaddata->flops+=2*2; // 1 128 bit operation

        // lines with L1 operations
        threaddata->flops+=70*2; // 1 128 bit operation

        // lines with L2 operations
        threaddata->flops+=0*2; // 1 128 bit operation

        // lines with L3 operations
        threaddata->flops+=0*2; // 1 128 bit operation

        // lines with RAM operations
        threaddata->flops+=1*2; // 1 128 bit operation
        threaddata->bytes=1*64; // 1 memory access

        threaddata->flops*=21;
        threaddata->bytes*=21;

	return EXIT_SUCCESS;
}

/**
 * assembler implementation of processor and memory stress test
 * uses SSE2 instruction set
 * optimized for Intel Nehalem/Westmere based Core i5/i7 and Xeon E3 with disabled Hyperthreading
 * @input - addrMem:   pointer to buffer
 * @return EXIT_SUCCESS
 */
int asm_work_nhm_corei_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_nhm_corei_sse2_1t(threaddata_t* threaddata)
{
	if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
		/* input: 
		 *   - addrMem -> rax
		 * register usage:
		 *   - rax:	stores original pointer to buffer, used to periodically reset other pointers
		 *   - rbx:	pointer to L1 buffer
		 *   - rcx:	pointer to L2 buffer
		 *   - rdx:	pointer to L3 buffer
		 *   - rdi:	pointer to RAM buffer
		 *   - r8:	counter for L2-pointer reset
		 *   - r9:	counter for L3-pointer reset
		 *   - r10:	counter for RAM-pointer reset
		 *   - r11:	temp register for initialization of SIMD-registers
		 *   - r12:	stores cacheline width as increment for buffer addresses
		 *   - r13:	stores address of shared variable that controls load level
                 *   - r14:	stores iteration counter
		 *   - mm*,xmm*:		data registers for SIMD instructions
		 */
	       __asm__ __volatile__(
		"mov %%rax, %%rax;" 	// store start address of buffer
		"mov %%rbx, %%r13;" 	// store address of shared variable that controls load level
                "mov %%rcx, %%r14;"	// store iteration counter
		"mov $64, %%r12;"	// increment after each cache/memory access
		//Initialize SSE-Registers for Addition
		"movapd 0(%%rax), %%xmm0;"
		"movapd 32(%%rax), %%xmm1;"
		"movapd 64(%%rax), %%xmm2;"
		"movapd 96(%%rax), %%xmm3;"
		"movapd 128(%%rax), %%xmm4;"
		"movapd 160(%%rax), %%xmm5;"
		"movapd 192(%%rax), %%xmm6;"
		"movapd 224(%%rax), %%xmm7;"
		"movapd 256(%%rax), %%xmm8;"
		"movapd 288(%%rax), %%xmm9;"
		"movapd 320(%%rax), %%xmm10;"
		"movapd 352(%%rax), %%xmm11;"
		"movapd 384(%%rax), %%xmm12;"
		"movapd 416(%%rax), %%xmm13;"
		//Initialize SSE-Registers for Transfer-Operations
		"movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
		"pinsrq $0, %%r11, %%xmm14;"
		"pinsrq $1, %%r11, %%xmm14;"
		"shl $4, %%r11;"
		"pinsrq $0, %%r11, %%xmm15;"
		"pinsrq $1, %%r11, %%xmm15;"
		"mov %%rax, %%rbx;" 	// address for L1-buffer 
		"mov %%rax, %%rcx;"     
		"add $32768, %%rcx;"	// address for L2-buffer
		"mov %%rax, %%rdx;"
		"add $262144, %%rdx;"	// address for L3-buffer
		"mov %%rax, %%rdi;"	 		
		"add $1572864, %%rdi;"	// address for RAM-buffer
		"movabs $0, %%r8;"	// reset-counter for L2-buffer with 0 cache lines accesses per loop (0 KB)
		"movabs $0, %%r9;"	// reset-counter for L3-buffer with 0 cache lines accesses per loop (0 KB)
		"movabs $78019, %%r10;"	// reset-counter for RAM-buffer with 21 cache lines accesses per loop (102399 KB)

                ".align 64;"     /* alignment in bytes */
                "_work_loop_nhm_corei_sse2_1t:"
		/****************************************************************************************************
		decode 0 			decode 1	 			decode 2	     	    */
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm11, %%xmm10;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm0, %%xmm13;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm9, %%xmm8;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm12, %%xmm11;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm1, %%xmm0;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm9, %%xmm8;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm4, %%xmm3;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm12, %%xmm11;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm7, %%xmm6;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm1, %%xmm0;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm4, %%xmm3;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm7, %%xmm6;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm2, %%xmm1;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm5, %%xmm4;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm8, %%xmm7;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm2, %%xmm1;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm11, %%xmm10;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm5, %%xmm4;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm0, %%xmm13;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm8, %%xmm7;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm11, %%xmm10;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm0, %%xmm13;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm9, %%xmm8;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm12, %%xmm11;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm1, %%xmm0;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm9, %%xmm8;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm4, %%xmm3;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm12, %%xmm11;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm7, %%xmm6;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm1, %%xmm0;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		//reset RAM counter
		"sub $1, %%r10;"
		"jnz _work_no_ram_reset_nhm_corei_sse2_1t;"
		"movabs $78019, %%r10;" 
		"mov %%rax, %%rdi;"
		"add $1572864, %%rdi;"
		"_work_no_ram_reset_nhm_corei_sse2_1t:"
                "inc %%r14;" // increment iteration counter
		"mov %%rax, %%rbx;"
                "testq $1, (%%r13);"
		"jnz _work_loop_nhm_corei_sse2_1t;"
                "movq %%r14, %%rax;" // restore iteration counter
                : "=a" (threaddata->iterations)
		: "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations) 
                : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
		);
	return EXIT_SUCCESS;
}




																																																																								int init_nhm_corei_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_corei_sse2_2t(threaddata_t* threaddata)
{
        unsigned long long addrMem = threaddata->addrMem;
	int i;
	for (i = 0; i<6670336; i++) *((double*)(addrMem + 8*i)) = i * 1.654738925401e-15;

        // lines with register operations
        threaddata->flops+=2*2; // 1 128 bit operation

        // lines with L1 operations
        threaddata->flops+=70*2; // 1 128 bit operation

        // lines with L2 operations
        threaddata->flops+=0*2; // 1 128 bit operation

        // lines with L3 operations
        threaddata->flops+=0*2; // 1 128 bit operation

        // lines with RAM operations
        threaddata->flops+=1*2; // 1 128 bit operation
        threaddata->bytes=1*64; // 1 memory access

        threaddata->flops*=10;
        threaddata->bytes*=10;

	return EXIT_SUCCESS;
}

/**
 * assembler implementation of processor and memory stress test
 * uses SSE2 instruction set
 * optimized for Intel Nehalem/Westmere based Core i5/i7 and Xeon E3 with enabled Hyperthreading
 * @input - addrMem:   pointer to buffer
 * @return EXIT_SUCCESS
 */
int asm_work_nhm_corei_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_nhm_corei_sse2_2t(threaddata_t* threaddata)
{
	if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
		/* input: 
		 *   - addrMem -> rax
		 * register usage:
		 *   - rax:	stores original pointer to buffer, used to periodically reset other pointers
		 *   - rbx:	pointer to L1 buffer
		 *   - rcx:	pointer to L2 buffer
		 *   - rdx:	pointer to L3 buffer
		 *   - rdi:	pointer to RAM buffer
		 *   - r8:	counter for L2-pointer reset
		 *   - r9:	counter for L3-pointer reset
		 *   - r10:	counter for RAM-pointer reset
		 *   - r11:	temp register for initialization of SIMD-registers
		 *   - r12:	stores cacheline width as increment for buffer addresses
		 *   - r13:	stores address of shared variable that controls load level
                 *   - r14:	stores iteration counter
		 *   - mm*,xmm*:		data registers for SIMD instructions
		 */
	       __asm__ __volatile__(
		"mov %%rax, %%rax;" 	// store start address of buffer
		"mov %%rbx, %%r13;" 	// store address of shared variable that controls load level
                "mov %%rcx, %%r14;"	// store iteration counter
		"mov $64, %%r12;"	// increment after each cache/memory access
		//Initialize SSE-Registers for Addition
		"movapd 0(%%rax), %%xmm0;"
		"movapd 32(%%rax), %%xmm1;"
		"movapd 64(%%rax), %%xmm2;"
		"movapd 96(%%rax), %%xmm3;"
		"movapd 128(%%rax), %%xmm4;"
		"movapd 160(%%rax), %%xmm5;"
		"movapd 192(%%rax), %%xmm6;"
		"movapd 224(%%rax), %%xmm7;"
		"movapd 256(%%rax), %%xmm8;"
		"movapd 288(%%rax), %%xmm9;"
		"movapd 320(%%rax), %%xmm10;"
		"movapd 352(%%rax), %%xmm11;"
		"movapd 384(%%rax), %%xmm12;"
		"movapd 416(%%rax), %%xmm13;"
		//Initialize SSE-Registers for Transfer-Operations
		"movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
		"pinsrq $0, %%r11, %%xmm14;"
		"pinsrq $1, %%r11, %%xmm14;"
		"shl $4, %%r11;"
		"pinsrq $0, %%r11, %%xmm15;"
		"pinsrq $1, %%r11, %%xmm15;"
		"mov %%rax, %%rbx;" 	// address for L1-buffer 
		"mov %%rax, %%rcx;"     
		"add $16384, %%rcx;"	// address for L2-buffer
		"mov %%rax, %%rdx;"
		"add $131072, %%rdx;"	// address for L3-buffer
		"mov %%rax, %%rdi;"	 		
		"add $786432, %%rdi;"	// address for RAM-buffer
		"movabs $0, %%r8;"	// reset-counter for L2-buffer with 0 cache lines accesses per loop (0 KB)
		"movabs $0, %%r9;"	// reset-counter for L3-buffer with 0 cache lines accesses per loop (0 KB)
		"movabs $81920, %%r10;"	// reset-counter for RAM-buffer with 10 cache lines accesses per loop (51200 KB)

                ".align 64;"     /* alignment in bytes */
                "_work_loop_nhm_corei_sse2_2t:"
		/****************************************************************************************************
		decode 0 			decode 1	 			decode 2	     	    */
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm11, %%xmm10;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm0, %%xmm13;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm9, %%xmm8;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm12, %%xmm11;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm1, %%xmm0;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm9, %%xmm8;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm4, %%xmm3;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm12, %%xmm11;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm7, %%xmm6;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm1, %%xmm0;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm4, %%xmm3;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm7, %%xmm6;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm2, %%xmm1;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		//reset RAM counter
		"sub $1, %%r10;"
		"jnz _work_no_ram_reset_nhm_corei_sse2_2t;"
		"movabs $81920, %%r10;" 
		"mov %%rax, %%rdi;"
		"add $786432, %%rdi;"
		"_work_no_ram_reset_nhm_corei_sse2_2t:"
                "inc %%r14;" // increment iteration counter
		"mov %%rax, %%rbx;"
                "testq $1, (%%r13);"
		"jnz _work_loop_nhm_corei_sse2_2t;"
                "movq %%r14, %%rax;" // restore iteration counter
                : "=a" (threaddata->iterations)
		: "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations) 
                : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
		);
	return EXIT_SUCCESS;
}




																																																														int init_nhm_xeonep_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_xeonep_sse2_1t(threaddata_t* threaddata)
{
        unsigned long long addrMem = threaddata->addrMem;
	int i;
	for (i = 0; i<13406208; i++) *((double*)(addrMem + 8*i)) = i * 1.654738925401e-15;

        // lines with register operations
        threaddata->flops+=2*2; // 1 128 bit operation

        // lines with L1 operations
        threaddata->flops+=60*2; // 1 128 bit operation

        // lines with L2 operations
        threaddata->flops+=0*2; // 1 128 bit operation

        // lines with L3 operations
        threaddata->flops+=0*2; // 1 128 bit operation

        // lines with RAM operations
        threaddata->flops+=1*2; // 1 128 bit operation
        threaddata->bytes=1*64; // 1 memory access

        threaddata->flops*=24;
        threaddata->bytes*=24;

	return EXIT_SUCCESS;
}

/**
 * assembler implementation of processor and memory stress test
 * uses SSE2 instruction set
 * optimized for Intel Nehalem/Westmere-EP based Core i7 and Xeon E5 with disabled Hyperthreading
 * @input - addrMem:   pointer to buffer
 * @return EXIT_SUCCESS
 */
int asm_work_nhm_xeonep_sse2_1t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_nhm_xeonep_sse2_1t(threaddata_t* threaddata)
{
	if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
		/* input: 
		 *   - addrMem -> rax
		 * register usage:
		 *   - rax:	stores original pointer to buffer, used to periodically reset other pointers
		 *   - rbx:	pointer to L1 buffer
		 *   - rcx:	pointer to L2 buffer
		 *   - rdx:	pointer to L3 buffer
		 *   - rdi:	pointer to RAM buffer
		 *   - r8:	counter for L2-pointer reset
		 *   - r9:	counter for L3-pointer reset
		 *   - r10:	counter for RAM-pointer reset
		 *   - r11:	temp register for initialization of SIMD-registers
		 *   - r12:	stores cacheline width as increment for buffer addresses
		 *   - r13:	stores address of shared variable that controls load level
                 *   - r14:	stores iteration counter
		 *   - mm*,xmm*:		data registers for SIMD instructions
		 */
	       __asm__ __volatile__(
		"mov %%rax, %%rax;" 	// store start address of buffer
		"mov %%rbx, %%r13;" 	// store address of shared variable that controls load level
                "mov %%rcx, %%r14;"	// store iteration counter
		"mov $64, %%r12;"	// increment after each cache/memory access
		//Initialize SSE-Registers for Addition
		"movapd 0(%%rax), %%xmm0;"
		"movapd 32(%%rax), %%xmm1;"
		"movapd 64(%%rax), %%xmm2;"
		"movapd 96(%%rax), %%xmm3;"
		"movapd 128(%%rax), %%xmm4;"
		"movapd 160(%%rax), %%xmm5;"
		"movapd 192(%%rax), %%xmm6;"
		"movapd 224(%%rax), %%xmm7;"
		"movapd 256(%%rax), %%xmm8;"
		"movapd 288(%%rax), %%xmm9;"
		"movapd 320(%%rax), %%xmm10;"
		"movapd 352(%%rax), %%xmm11;"
		"movapd 384(%%rax), %%xmm12;"
		"movapd 416(%%rax), %%xmm13;"
		//Initialize SSE-Registers for Transfer-Operations
		"movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
		"pinsrq $0, %%r11, %%xmm14;"
		"pinsrq $1, %%r11, %%xmm14;"
		"shl $4, %%r11;"
		"pinsrq $0, %%r11, %%xmm15;"
		"pinsrq $1, %%r11, %%xmm15;"
		"mov %%rax, %%rbx;" 	// address for L1-buffer 
		"mov %%rax, %%rcx;"     
		"add $32768, %%rcx;"	// address for L2-buffer
		"mov %%rax, %%rdx;"
		"add $262144, %%rdx;"	// address for L3-buffer
		"mov %%rax, %%rdi;"	 		
		"add $2097152, %%rdi;"	// address for RAM-buffer
		"movabs $0, %%r8;"	// reset-counter for L2-buffer with 0 cache lines accesses per loop (0 KB)
		"movabs $0, %%r9;"	// reset-counter for L3-buffer with 0 cache lines accesses per loop (0 KB)
		"movabs $68266, %%r10;"	// reset-counter for RAM-buffer with 24 cache lines accesses per loop (102399 KB)

                ".align 64;"     /* alignment in bytes */
                "_work_loop_nhm_xeonep_sse2_1t:"
		/****************************************************************************************************
		decode 0 			decode 1	 			decode 2	     	    */
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		//reset RAM counter
		"sub $1, %%r10;"
		"jnz _work_no_ram_reset_nhm_xeonep_sse2_1t;"
		"movabs $68266, %%r10;" 
		"mov %%rax, %%rdi;"
		"add $2097152, %%rdi;"
		"_work_no_ram_reset_nhm_xeonep_sse2_1t:"
                "inc %%r14;" // increment iteration counter
		"mov %%rax, %%rbx;"
                "testq $1, (%%r13);"
		"jnz _work_loop_nhm_xeonep_sse2_1t;"
                "movq %%r14, %%rax;" // restore iteration counter
                : "=a" (threaddata->iterations)
		: "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations) 
                : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
		);
	return EXIT_SUCCESS;
}




																																																														int init_nhm_xeonep_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int init_nhm_xeonep_sse2_2t(threaddata_t* threaddata)
{
        unsigned long long addrMem = threaddata->addrMem;
	int i;
	for (i = 0; i<6703104; i++) *((double*)(addrMem + 8*i)) = i * 1.654738925401e-15;

        // lines with register operations
        threaddata->flops+=2*2; // 1 128 bit operation

        // lines with L1 operations
        threaddata->flops+=60*2; // 1 128 bit operation

        // lines with L2 operations
        threaddata->flops+=0*2; // 1 128 bit operation

        // lines with L3 operations
        threaddata->flops+=0*2; // 1 128 bit operation

        // lines with RAM operations
        threaddata->flops+=1*2; // 1 128 bit operation
        threaddata->bytes=1*64; // 1 memory access

        threaddata->flops*=12;
        threaddata->bytes*=12;

	return EXIT_SUCCESS;
}

/**
 * assembler implementation of processor and memory stress test
 * uses SSE2 instruction set
 * optimized for Intel Nehalem/Westmere-EP based Core i7 and Xeon E5 with enabled Hyperthreading
 * @input - addrMem:   pointer to buffer
 * @return EXIT_SUCCESS
 */
int asm_work_nhm_xeonep_sse2_2t(threaddata_t* threaddata) __attribute__((noinline));
int asm_work_nhm_xeonep_sse2_2t(threaddata_t* threaddata)
{
	if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;
		/* input: 
		 *   - addrMem -> rax
		 * register usage:
		 *   - rax:	stores original pointer to buffer, used to periodically reset other pointers
		 *   - rbx:	pointer to L1 buffer
		 *   - rcx:	pointer to L2 buffer
		 *   - rdx:	pointer to L3 buffer
		 *   - rdi:	pointer to RAM buffer
		 *   - r8:	counter for L2-pointer reset
		 *   - r9:	counter for L3-pointer reset
		 *   - r10:	counter for RAM-pointer reset
		 *   - r11:	temp register for initialization of SIMD-registers
		 *   - r12:	stores cacheline width as increment for buffer addresses
		 *   - r13:	stores address of shared variable that controls load level
                 *   - r14:	stores iteration counter
		 *   - mm*,xmm*:		data registers for SIMD instructions
		 */
	       __asm__ __volatile__(
		"mov %%rax, %%rax;" 	// store start address of buffer
		"mov %%rbx, %%r13;" 	// store address of shared variable that controls load level
                "mov %%rcx, %%r14;"	// store iteration counter
		"mov $64, %%r12;"	// increment after each cache/memory access
		//Initialize SSE-Registers for Addition
		"movapd 0(%%rax), %%xmm0;"
		"movapd 32(%%rax), %%xmm1;"
		"movapd 64(%%rax), %%xmm2;"
		"movapd 96(%%rax), %%xmm3;"
		"movapd 128(%%rax), %%xmm4;"
		"movapd 160(%%rax), %%xmm5;"
		"movapd 192(%%rax), %%xmm6;"
		"movapd 224(%%rax), %%xmm7;"
		"movapd 256(%%rax), %%xmm8;"
		"movapd 288(%%rax), %%xmm9;"
		"movapd 320(%%rax), %%xmm10;"
		"movapd 352(%%rax), %%xmm11;"
		"movapd 384(%%rax), %%xmm12;"
		"movapd 416(%%rax), %%xmm13;"
		//Initialize SSE-Registers for Transfer-Operations
		"movabs $0x0F0F0F0F0F0F0F0F, %%r11;"
		"pinsrq $0, %%r11, %%xmm14;"
		"pinsrq $1, %%r11, %%xmm14;"
		"shl $4, %%r11;"
		"pinsrq $0, %%r11, %%xmm15;"
		"pinsrq $1, %%r11, %%xmm15;"
		"mov %%rax, %%rbx;" 	// address for L1-buffer 
		"mov %%rax, %%rcx;"     
		"add $16384, %%rcx;"	// address for L2-buffer
		"mov %%rax, %%rdx;"
		"add $131072, %%rdx;"	// address for L3-buffer
		"mov %%rax, %%rdi;"	 		
		"add $1048576, %%rdi;"	// address for RAM-buffer
		"movabs $0, %%r8;"	// reset-counter for L2-buffer with 0 cache lines accesses per loop (0 KB)
		"movabs $0, %%r9;"	// reset-counter for L3-buffer with 0 cache lines accesses per loop (0 KB)
		"movabs $68266, %%r10;"	// reset-counter for RAM-buffer with 12 cache lines accesses per loop (51199 KB)

                ".align 64;"     /* alignment in bytes */
                "_work_loop_nhm_xeonep_sse2_2t:"
		/****************************************************************************************************
		decode 0 			decode 1	 			decode 2	     	    */
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm3, %%xmm2;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm6, %%xmm5;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		mov %%rax, %%rbx;	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	prefetcht2 (%%rdi);			add %%r12, %%rdi;  	"	// RAM prefetch
		"addpd %%xmm10, %%xmm9;  						movdqa %%xmm15, %%xmm14;"	// REG ops only
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd %%xmm13, %%xmm12;  						movdqa %%xmm14, %%xmm15;"	// REG ops only
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm1;	movapd %%xmm1, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm2;	movapd %%xmm2, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm3;	movapd %%xmm3, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm4;	movapd %%xmm4, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm5;	movapd %%xmm5, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm6;	movapd %%xmm6, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm7;	movapd %%xmm7, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm8;	movapd %%xmm8, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm9;	movapd %%xmm9, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm10;	movapd %%xmm10, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm11;	movapd %%xmm11, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm12;	movapd %%xmm12, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm13;	movapd %%xmm13, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		"addpd 32(%%rbx), %%xmm0;	movapd %%xmm0, 64(%%rbx);		add %%r12, %%rbx;  	"	// L1 load, L1 store
		//reset RAM counter
		"sub $1, %%r10;"
		"jnz _work_no_ram_reset_nhm_xeonep_sse2_2t;"
		"movabs $68266, %%r10;" 
		"mov %%rax, %%rdi;"
		"add $1048576, %%rdi;"
		"_work_no_ram_reset_nhm_xeonep_sse2_2t:"
                "inc %%r14;" // increment iteration counter
		"mov %%rax, %%rbx;"
                "testq $1, (%%r13);"
		"jnz _work_loop_nhm_xeonep_sse2_2t;"
                "movq %%r14, %%rax;" // restore iteration counter
                : "=a" (threaddata->iterations)
		: "a"(threaddata->addrMem), "b"(threaddata->addrHigh), "c" (threaddata->iterations) 
                : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
		);
	return EXIT_SUCCESS;
}
