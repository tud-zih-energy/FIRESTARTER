/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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
 * along with this program.  If not, see <http://www.gnu.org/licenses/\>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

#include <chrono>
#include <thread>
#include <type_traits>

#ifdef _MSC_VER
#include <array>
#include <intrin.h>
#endif

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

using namespace firestarter::environment::x86::payload;

void X86Payload::lowLoadFunction(volatile unsigned long long *addrHigh,
                                 unsigned long long period) {
  int nap;
#ifdef _MSC_VER
  std::array<int, 4> cpuid;
#endif

  nap = period / 100;
#ifndef _MSC_VER
  __asm__ __volatile__("mfence;"
                       "cpuid;" ::
                           : "eax", "ebx", "ecx", "edx");
#else
  _mm_mfence();
  __cpuid(cpuid.data(), 0);
#endif
  // while signal low load
  while (*addrHigh == LOAD_LOW) {
#ifndef _MSC_VER
    __asm__ __volatile__("mfence;"
                         "cpuid;" ::
                             : "eax", "ebx", "ecx", "edx");
#else
    _mm_mfence();
    __cpuid(cpuid.data(), 0);
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(nap));
#ifndef _MSC_VER
    __asm__ __volatile__("mfence;"
                         "cpuid;" ::
                             : "eax", "ebx", "ecx", "edx");
#else
    _mm_mfence();
    __cpuid(cpuid.data(), 0);
#endif
  }
}

void X86Payload::init(unsigned long long *memoryAddr,
                      unsigned long long bufferSize, double firstValue,
                      double lastValue) {
  unsigned long long i = 0;

  for (; i < INIT_BLOCKSIZE; i++)
    *((double *)(memoryAddr + i)) = 0.25 + (double)i * 8.0 * firstValue;
  for (; i <= bufferSize - INIT_BLOCKSIZE; i += INIT_BLOCKSIZE)
    std::memcpy(memoryAddr + i, memoryAddr + i - INIT_BLOCKSIZE,
                sizeof(unsigned long long) * INIT_BLOCKSIZE);
  for (; i < bufferSize; i++)
    *((double *)(memoryAddr + i)) = 0.25 + (double)i * 8.0 * lastValue;
}

unsigned long long
X86Payload::highLoadFunction(unsigned long long *addrMem,
                             volatile unsigned long long *addrHigh,
                             unsigned long long iterations) {
  return this->loadFunction(addrMem, addrHigh, iterations);
}

// add MM regs to dirty regs
template <>
void X86Payload::emitErrorDetectionCode<asmjit::x86::Ymm>(
    asmjit::x86::Builder &cb, asmjit::x86::Mm iter_reg,
    asmjit::x86::Gp temp_reg, asmjit::x86::Gp temp_reg2) {
  // static_assert(std::is_base_of<asmjit::x86::Vec, Vec>::value, "Vec must be
  // of asmjit::asmjit::x86::Vec");
  assert(((iter_reg == asmjit::x86::mm0),
          "iter_reg must be asmjit::asmjit::x86::mm0"));

  // TODO: implement for xmm and zmm
  // static_assert(std::is_same<asmjit::x86::Ymm, Vec>::value, "Not implemented
  // for any other type than asmjit::asmjit::x86::Ymm");

  // do the error detection every 1024 (2**10) iterations
  // check if lower 10 bits eq 0
  auto SkipErrorDetection = cb.newLabel();

  cb.movq(temp_reg, iter_reg);
  cb.and_(temp_reg, asmjit::Imm(0x3ff));
  cb.test(temp_reg, asmjit::Imm(0));
  cb.jnz(SkipErrorDetection);

  // ONLY IF ZMM ?
#if 0
			// 0. move mm0 (iter_reg) to temp_reg
			cb.movq(temp_reg, iter_reg);
#endif
  cb.mov(temp_reg, asmjit::Imm(0xffffffff));

  // 1. use mm to backup vector registers 0
  cb.movq(temp_reg2, asmjit::x86::xmm0);
  cb.movq(asmjit::x86::Mm(7), temp_reg2);
  cb.crc32(temp_reg, temp_reg2);
  cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
  cb.movq(temp_reg2, asmjit::x86::xmm0);
  cb.movq(asmjit::x86::Mm(6), temp_reg2);
  cb.crc32(temp_reg, temp_reg2);

  cb.vextractf128(asmjit::x86::xmm0, asmjit::x86::ymm0, asmjit::Imm(1));

  cb.movq(temp_reg2, asmjit::x86::xmm0);
  cb.movq(asmjit::x86::Mm(5), temp_reg2);
  cb.crc32(temp_reg, temp_reg2);
  cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
  cb.movq(temp_reg2, asmjit::x86::xmm0);
  cb.movq(asmjit::x86::Mm(4), temp_reg2);
  cb.crc32(temp_reg, temp_reg2);

  // 1. calculate the hash
  for (int i = 1; i < (int)this->registerCount(); i++) {
    // mov vector[i] to vector[0]
    cb.vmovapd(asmjit::x86::ymm0, asmjit::x86::Ymm(i));

    // calculate hash
    cb.movq(temp_reg2, asmjit::x86::xmm0);
    cb.crc32(temp_reg, temp_reg2);
    cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
    cb.movq(temp_reg2, asmjit::x86::xmm0);
    cb.crc32(temp_reg, temp_reg2);

    cb.vextractf128(asmjit::x86::xmm0, asmjit::x86::ymm0, asmjit::Imm(1));

    cb.movq(temp_reg2, asmjit::x86::xmm0);
    cb.crc32(temp_reg, temp_reg2);
    cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
    cb.movq(temp_reg2, asmjit::x86::xmm0);
    cb.crc32(temp_reg, temp_reg2);
  }

  // N-1. restore vector register 0 from backup
  cb.movq(temp_reg2, asmjit::x86::Mm(5));
  cb.movq(asmjit::x86::xmm0, temp_reg2);
  cb.movq(temp_reg2, asmjit::x86::Mm(4));
  cb.pinsrq(asmjit::x86::xmm0, temp_reg2, asmjit::Imm(1));

  cb.vinsertf128(asmjit::x86::ymm0, asmjit::x86::ymm0, asmjit::x86::xmm0,
                 asmjit::Imm(1));

  cb.movq(temp_reg2, asmjit::x86::Mm(7));
  cb.movq(asmjit::x86::xmm0, temp_reg2);
  cb.movq(temp_reg2, asmjit::x86::Mm(6));
  cb.pinsrq(asmjit::x86::xmm0, temp_reg2, asmjit::Imm(1));

  // before starting the communication, backup rax, rbx, rcx and rdx to mm[0:3]
  cb.movq(asmjit::x86::Mm(7), asmjit::x86::rax);
  cb.movq(asmjit::x86::Mm(6), asmjit::x86::rbx);
  cb.movq(asmjit::x86::Mm(5), asmjit::x86::rcx);
  cb.movq(asmjit::x86::Mm(4), asmjit::x86::rdx);

  // communication

  // restore rax, rbx, rcx and rdx from mm[0:3]
  cb.movq(asmjit::x86::rax, asmjit::x86::Mm(7));
  cb.movq(asmjit::x86::rbx, asmjit::x86::Mm(6));
  cb.movq(asmjit::x86::rcx, asmjit::x86::Mm(5));
  cb.movq(asmjit::x86::rdx, asmjit::x86::Mm(4));

  // N. move temp_reg back to iter_reg
  cb.movq(iter_reg, temp_reg);

  cb.bind(SkipErrorDetection);
}
