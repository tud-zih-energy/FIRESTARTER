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
// zmm31 is used for backup if VectorReg is of type asmjit::x86::Zmm
template <class IterReg, class VectorReg>
void X86Payload::emitErrorDetectionCode(asmjit::x86::Builder &cb,
                                        IterReg iter_reg,
                                        asmjit::x86::Gpq addrHigh_reg,
                                        asmjit::x86::Gpq pointer_reg,
                                        asmjit::x86::Gpq temp_reg,
                                        asmjit::x86::Gpq temp_reg2) {
  // we don't want anything to break... so we use asserts for everything that
  // could break it
  static_assert(std::is_base_of<asmjit::x86::Vec, VectorReg>::value,
                "VectorReg must be of asmjit::asmjit::x86::Vec");
  static_assert(std::is_same<asmjit::x86::Xmm, VectorReg>::value ||
                    std::is_same<asmjit::x86::Ymm, VectorReg>::value ||
                    std::is_same<asmjit::x86::Zmm, VectorReg>::value,
                "VectorReg ist not of any supported type");
  static_assert(std::is_same<asmjit::x86::Mm, IterReg>::value ||
                    std::is_same<asmjit::x86::Gpq, IterReg>::value,
                "IterReg is not of any supported type");

  if constexpr (std::is_same<asmjit::x86::Mm, IterReg>::value) {
    assert((iter_reg == asmjit::x86::mm0, "iter_reg must be mm0"));
  }

  assert((iter_reg != temp_reg, "iter_reg must be != temp_reg"));
  assert((temp_reg != temp_reg2, "temp_reg must be != temp_reg2"));
  assert((temp_reg != addrHigh_reg, "temp_reg must be != addrHigh_reg"));
  assert((temp_reg != pointer_reg, "temp_reg must be != pointer_reg"));

  assert((iter_reg != asmjit::x86::r8, "iter_reg must be != r8"));
  assert((iter_reg != asmjit::x86::r9, "iter_reg must be != r9"));
  assert((iter_reg != asmjit::x86::rax, "iter_reg must be != rax"));
  assert((iter_reg != asmjit::x86::rbx, "iter_reg must be != rbx"));
  assert((iter_reg != asmjit::x86::rcx, "iter_reg must be != rcx"));
  assert((iter_reg != asmjit::x86::rdx, "iter_reg must be != rdx"));

  assert((temp_reg != asmjit::x86::r8, "temp_reg must be != r8"));
  assert((temp_reg != asmjit::x86::r9, "temp_reg must be != r9"));
  assert((temp_reg != asmjit::x86::rax, "temp_reg must be != rax"));
  assert((temp_reg != asmjit::x86::rbx, "temp_reg must be != rbx"));
  assert((temp_reg != asmjit::x86::rcx, "temp_reg must be != rcx"));
  assert((temp_reg != asmjit::x86::rdx, "temp_reg must be != rdx"));

  assert((temp_reg2 != asmjit::x86::r8, "temp_reg2 must be != r8"));
  assert((temp_reg2 != asmjit::x86::r9, "temp_reg2 must be != r9"));
  assert((temp_reg2 != asmjit::x86::rax, "temp_reg2 must be != rax"));
  assert((temp_reg2 != asmjit::x86::rbx, "temp_reg2 must be != rbx"));
  assert((temp_reg2 != asmjit::x86::rcx, "temp_reg2 must be != rcx"));
  assert((temp_reg2 != asmjit::x86::rdx, "temp_reg2 must be != rdx"));

  assert((addrHigh_reg != asmjit::x86::r8, "addrHigh_reg must be != r8"));
  assert((addrHigh_reg != asmjit::x86::r9, "addrHigh_reg must be != r9"));
  assert((addrHigh_reg != asmjit::x86::rax, "addrHigh_reg must be != rax"));
  assert((addrHigh_reg != asmjit::x86::rbx, "addrHigh_reg must be != rbx"));
  assert((addrHigh_reg != asmjit::x86::rcx, "addrHigh_reg must be != rcx"));
  assert((addrHigh_reg != asmjit::x86::rdx, "addrHigh_reg must be != rdx"));

  auto SkipErrorDetection = cb.newLabel();

  if constexpr (std::is_same<asmjit::x86::Mm, IterReg>::value) {
    cb.movq(temp_reg, iter_reg);
  } else {
    cb.mov(temp_reg, iter_reg);
  }
  // round about 50-100 Hz
  // more or less, but this isn't really that relevant
  cb.and_(temp_reg, asmjit::Imm(0x3fff));
  cb.test(temp_reg, temp_reg);
  cb.jnz(SkipErrorDetection);

  cb.mov(temp_reg, asmjit::Imm(0xffffffff));

  int registerCount = (int)this->registerCount();

  // Create a backup of VectorReg(0)
  if constexpr (std::is_same<asmjit::x86::Xmm, VectorReg>::value) {
    cb.movq(temp_reg2, asmjit::x86::xmm0);
    cb.push(temp_reg2);
    cb.crc32(temp_reg, temp_reg2);
    cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
    cb.movq(temp_reg2, asmjit::x86::xmm0);
    cb.push(temp_reg2);
    cb.crc32(temp_reg, temp_reg2);

  } else if constexpr (std::is_same<asmjit::x86::Ymm, VectorReg>::value &&
                       std::is_same<asmjit::x86::Mm, IterReg>::value) {
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
  } else if constexpr (std::is_same<asmjit::x86::Zmm, VectorReg>::value &&
                       std::is_same<asmjit::x86::Mm, IterReg>::value) {
    // We use vector registers zmm31 for our backup
    cb.vmovapd(asmjit::x86::zmm31, asmjit::x86::zmm0);
    registerCount--;
  }

  // Calculate the hash of the remaining VectorReg
  // use VectorReg(0) as a temporary place to unpack values
  for (int i = 1; i < registerCount; i++) {
    if constexpr (std::is_same<asmjit::x86::Xmm, VectorReg>::value) {
      cb.vmovapd(asmjit::x86::xmm0, asmjit::x86::Xmm(i));

      cb.movq(temp_reg2, asmjit::x86::xmm0);
      cb.crc32(temp_reg, temp_reg2);
      cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      cb.movq(temp_reg2, asmjit::x86::xmm0);
      cb.crc32(temp_reg, temp_reg2);
    } else if constexpr (std::is_same<asmjit::x86::Ymm, VectorReg>::value) {
      cb.vmovapd(asmjit::x86::ymm0, asmjit::x86::Ymm(i));

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
    } else if constexpr (std::is_same<asmjit::x86::Zmm, VectorReg>::value) {
      cb.vmovapd(asmjit::x86::ymm0, asmjit::x86::Ymm(i));

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

      cb.vextractf32x4(asmjit::x86::xmm0, asmjit::x86::Zmm(i), asmjit::Imm(2));

      cb.movq(temp_reg2, asmjit::x86::xmm0);
      cb.crc32(temp_reg, temp_reg2);
      cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      cb.movq(temp_reg2, asmjit::x86::xmm0);
      cb.crc32(temp_reg, temp_reg2);

      cb.vextractf32x4(asmjit::x86::xmm0, asmjit::x86::Zmm(i), asmjit::Imm(3));

      cb.movq(temp_reg2, asmjit::x86::xmm0);
      cb.crc32(temp_reg, temp_reg2);
      cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      cb.movq(temp_reg2, asmjit::x86::xmm0);
      cb.crc32(temp_reg, temp_reg2);
    }
  }

  // Restore VectorReg(0) from backup
  if constexpr (std::is_same<asmjit::x86::Xmm, VectorReg>::value) {
    cb.pop(temp_reg2);
    cb.movq(asmjit::x86::xmm0, temp_reg2);
    cb.movlhps(asmjit::x86::xmm0, asmjit::x86::xmm0);
    cb.pop(temp_reg2);
    cb.pinsrw(asmjit::x86::xmm0, temp_reg2.r32(), asmjit::Imm(0));
    cb.shr(temp_reg2, asmjit::Imm(32));
    cb.movd(temp_reg2.r32(), asmjit::x86::Mm(7));
    cb.pinsrw(asmjit::x86::xmm0, temp_reg2.r32(), asmjit::Imm(1));
  } else if constexpr (std::is_same<asmjit::x86::Ymm, VectorReg>::value &&
                       std::is_same<asmjit::x86::Mm, IterReg>::value) {
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
  } else if constexpr (std::is_same<asmjit::x86::Zmm, VectorReg>::value &&
                       std::is_same<asmjit::x86::Mm, IterReg>::value) {
    // We use vector registers zmm31 for our backup
    cb.vmovapd(asmjit::x86::zmm0, asmjit::x86::zmm31);
  }

  // before starting the communication, backup r8, r9, rax, rbx, rcx and rdx
  if constexpr (std::is_same<asmjit::x86::Mm, IterReg>::value) {
    cb.movq(asmjit::x86::Mm(7), asmjit::x86::rax);
    cb.movq(asmjit::x86::Mm(6), asmjit::x86::rbx);
    cb.movq(asmjit::x86::Mm(5), asmjit::x86::rcx);
    cb.movq(asmjit::x86::Mm(4), asmjit::x86::rdx);
    cb.movq(asmjit::x86::Mm(3), asmjit::x86::r8);
    cb.movq(asmjit::x86::Mm(2), asmjit::x86::r9);
  } else {
    cb.push(asmjit::x86::rax);
    cb.push(asmjit::x86::rbx);
    cb.push(asmjit::x86::rcx);
    cb.push(asmjit::x86::rdx);
    cb.push(asmjit::x86::r8);
    cb.push(asmjit::x86::r9);
  }

  // do the actual communication
  // temp_reg contains our hash

  // save the pointer_reg. it might be any of r8, r9, rax, rbx, rcx or rdx
  cb.mov(temp_reg2, pointer_reg);

  // Don't touch me!
  // This sychronization and communication works even if the threads run at
  // different (changing) speed, with just one "lock cmpxchg16b" Brought to you
  // by a few hours of headache for two people.
  auto communication = [&](auto offset) {
    // communication
    cb.mov(asmjit::x86::r8, asmjit::x86::ptr_64(temp_reg2, offset));

    // temp data
    cb.mov(asmjit::x86::r9, temp_reg2);
    cb.add(asmjit::x86::r9, asmjit::Imm(offset + 8));

    cb.mov(asmjit::x86::rdx, asmjit::x86::ptr_64(asmjit::x86::r9, 0));
    cb.mov(asmjit::x86::rax, asmjit::x86::ptr_64(asmjit::x86::r9, 8));

    auto L0 = cb.newLabel();
    cb.bind(L0);

    cb.lock();
    cb.cmpxchg16b(asmjit::x86::ptr(asmjit::x86::r8));

    auto L1 = cb.newLabel();
    cb.jnz(L1);

    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 0), asmjit::x86::rcx);
    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 8), asmjit::x86::rbx);
    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 16), asmjit::Imm(0));
    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 24), asmjit::Imm(0));

    cb.mov(asmjit::x86::rax, asmjit::Imm(2));

    auto L6 = cb.newLabel();
    cb.jmp(L6);

    cb.bind(L1);

    cb.cmp(asmjit::x86::rcx, asmjit::x86::rdx);

    auto L2 = cb.newLabel();
    cb.jle(L2);

    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 0), asmjit::x86::rcx);
    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 8), asmjit::x86::rbx);

    cb.jmp(L0);

    cb.bind(L2);

    auto L3 = cb.newLabel();

    cb.cmp(asmjit::x86::ptr_64(asmjit::x86::r9, 16), asmjit::Imm(0));
    cb.jne(L3);
    cb.cmp(asmjit::x86::ptr_64(asmjit::x86::r9, 24), asmjit::Imm(0));
    cb.jne(L3);

    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 16), asmjit::x86::rdx);
    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 24), asmjit::x86::rax);

    cb.bind(L3);

    cb.cmp(asmjit::x86::rcx, asmjit::x86::ptr_64(asmjit::x86::r9, 16));
    cb.mov(asmjit::x86::rax, asmjit::Imm(4));
    cb.jne(L6);

    cb.cmp(asmjit::x86::rbx, asmjit::x86::ptr_64(asmjit::x86::r9, 24));
    auto L4 = cb.newLabel();
    cb.jne(L4);

    cb.mov(asmjit::x86::rax, asmjit::Imm(0));

    auto L5 = cb.newLabel();
    cb.jmp(L5);

    cb.bind(L4);

    cb.mov(asmjit::x86::rax, asmjit::Imm(1));

    cb.bind(L5);

    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 16), asmjit::Imm(0));
    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 24), asmjit::Imm(0));

    cb.bind(L6);

    // if check failed
    cb.cmp(asmjit::x86::rax, asmjit::Imm(1));
    auto L7 = cb.newLabel();
    cb.jne(L7);

    // write the error flag
    cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 32), asmjit::Imm(1));

    // stop the execution after some time
    cb.mov(asmjit::x86::ptr_64(addrHigh_reg), asmjit::Imm(LOAD_STOP));
    cb.mfence();

    cb.bind(L7);

    auto L9 = cb.newLabel();
    cb.jmp(L9);
  };

  // left communication
  // move hash
  cb.mov(asmjit::x86::rbx, temp_reg);
  // move iterations counter
  if constexpr (std::is_same<asmjit::x86::Mm, IterReg>::value) {
    cb.movq(asmjit::x86::rcx, iter_reg);
  } else {
    cb.mov(asmjit::x86::rcx, iter_reg);
  }

  communication(-128);

  // right communication
  // move hash
  cb.mov(asmjit::x86::rbx, temp_reg);
  // move iterations counter
  if constexpr (std::is_same<asmjit::x86::Mm, IterReg>::value) {
    cb.movq(asmjit::x86::rcx, iter_reg);
  } else {
    cb.mov(asmjit::x86::rcx, iter_reg);
  }

  communication(-64);

  // restore r8, r9, rax, rbx, rcx and rdx
  if constexpr (std::is_same<asmjit::x86::Mm, IterReg>::value) {
    cb.movq(asmjit::x86::rax, asmjit::x86::Mm(7));
    cb.movq(asmjit::x86::rbx, asmjit::x86::Mm(6));
    cb.movq(asmjit::x86::rcx, asmjit::x86::Mm(5));
    cb.movq(asmjit::x86::rdx, asmjit::x86::Mm(4));
    cb.movq(asmjit::x86::r8, asmjit::x86::Mm(3));
    cb.movq(asmjit::x86::r9, asmjit::x86::Mm(2));
  } else {
    cb.pop(asmjit::x86::r9);
    cb.pop(asmjit::x86::r8);
    cb.pop(asmjit::x86::rdx);
    cb.pop(asmjit::x86::rcx);
    cb.pop(asmjit::x86::rbx);
    cb.pop(asmjit::x86::rax);
  }

  cb.bind(SkipErrorDetection);
}

template void
X86Payload::emitErrorDetectionCode<asmjit::x86::Gpq, asmjit::x86::Xmm>(
    asmjit::x86::Builder &cb, asmjit::x86::Gpq iter_reg,
    asmjit::x86::Gpq addrHigh_reg, asmjit::x86::Gpq pointer_reg,
    asmjit::x86::Gpq temp_reg, asmjit::x86::Gpq temp_reg2);
template void
X86Payload::emitErrorDetectionCode<asmjit::x86::Gpq, asmjit::x86::Ymm>(
    asmjit::x86::Builder &cb, asmjit::x86::Gpq iter_reg,
    asmjit::x86::Gpq addrHigh_reg, asmjit::x86::Gpq pointer_reg,
    asmjit::x86::Gpq temp_reg, asmjit::x86::Gpq temp_reg2);

template void
X86Payload::emitErrorDetectionCode<asmjit::x86::Mm, asmjit::x86::Ymm>(
    asmjit::x86::Builder &cb, asmjit::x86::Mm iter_reg,
    asmjit::x86::Gpq addrHigh_reg, asmjit::x86::Gpq pointer_reg,
    asmjit::x86::Gpq temp_reg, asmjit::x86::Gpq temp_reg2);
template void
X86Payload::emitErrorDetectionCode<asmjit::x86::Mm, asmjit::x86::Zmm>(
    asmjit::x86::Builder &cb, asmjit::x86::Mm iter_reg,
    asmjit::x86::Gpq addrHigh_reg, asmjit::x86::Gpq pointer_reg,
    asmjit::x86::Gpq temp_reg, asmjit::x86::Gpq temp_reg2);
