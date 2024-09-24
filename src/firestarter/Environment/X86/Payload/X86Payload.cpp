/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
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

void X86Payload::lowLoadFunction(volatile uint64_t* addrHigh, uint64_t period) {
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

void X86Payload::init(uint64_t* memoryAddr, uint64_t bufferSize, double firstValue, double lastValue) {
  uint64_t i = 0;

  for (; i < INIT_BLOCKSIZE; i++)
    *((double*)(memoryAddr + i)) = 0.25 + (double)i * 8.0 * firstValue;
  for (; i <= bufferSize - INIT_BLOCKSIZE; i += INIT_BLOCKSIZE)
    std::memcpy(memoryAddr + i, memoryAddr + i - INIT_BLOCKSIZE, sizeof(uint64_t) * INIT_BLOCKSIZE);
  for (; i < bufferSize; i++)
    *((double*)(memoryAddr + i)) = 0.25 + (double)i * 8.0 * lastValue;
}

uint64_t X86Payload::highLoadFunction(uint64_t* addrMem, volatile uint64_t* addrHigh, uint64_t iterations) {
  return this->LoadFunction(addrMem, addrHigh, iterations);
}

// add MM regs to dirty regs
// zmm31 is used for backup if VectorReg is of type asmjit::x86::Zmm
template <class IterRegT, class VectorRegT>
void X86Payload::emitErrorDetectionCode(asmjit::x86::Builder& Cb, IterRegT IterReg, asmjit::x86::Gpq addrHigh_reg,
                                        asmjit::x86::Gpq pointer_reg, asmjit::x86::Gpq temp_reg,
                                        asmjit::x86::Gpq temp_reg2) {
  // we don't want anything to break... so we use asserts for everything that
  // could break it
  static_assert(std::is_base_of<asmjit::x86::Vec, VectorRegT>::value, "VectorReg must be of asmjit::asmjit::x86::Vec");
  static_assert(std::is_same<asmjit::x86::Xmm, VectorRegT>::value ||
                    std::is_same<asmjit::x86::Ymm, VectorRegT>::value ||
                    std::is_same<asmjit::x86::Zmm, VectorRegT>::value,
                "VectorReg ist not of any supported type");
  static_assert(std::is_same<asmjit::x86::Mm, IterRegT>::value || std::is_same<asmjit::x86::Gpq, IterRegT>::value,
                "IterReg is not of any supported type");

  if constexpr (std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    assert((IterReg == asmjit::x86::mm0, "iter_reg must be mm0"));
  }

  assert((IterReg != temp_reg, "iter_reg must be != temp_reg"));
  assert((temp_reg != temp_reg2, "temp_reg must be != temp_reg2"));
  assert((temp_reg != addrHigh_reg, "temp_reg must be != addrHigh_reg"));
  assert((temp_reg != pointer_reg, "temp_reg must be != pointer_reg"));

  assert((IterReg != asmjit::x86::r8, "iter_reg must be != r8"));
  assert((IterReg != asmjit::x86::r9, "iter_reg must be != r9"));
  assert((IterReg != asmjit::x86::rax, "iter_reg must be != rax"));
  assert((IterReg != asmjit::x86::rbx, "iter_reg must be != rbx"));
  assert((IterReg != asmjit::x86::rcx, "iter_reg must be != rcx"));
  assert((IterReg != asmjit::x86::rdx, "iter_reg must be != rdx"));

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

  auto SkipErrorDetection = Cb.newLabel();

  if constexpr (std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    Cb.movq(temp_reg, IterReg);
  } else {
    Cb.mov(temp_reg, IterReg);
  }
  // round about 50-100 Hz
  // more or less, but this isn't really that relevant
  Cb.and_(temp_reg, asmjit::Imm(0x3fff));
  Cb.test(temp_reg, temp_reg);
  Cb.jnz(SkipErrorDetection);

  Cb.mov(temp_reg, asmjit::Imm(0xffffffff));

  int registerCount = (int)this->registerCount();

  // Create a backup of VectorReg(0)
  if constexpr (std::is_same<asmjit::x86::Xmm, VectorRegT>::value) {
    Cb.movq(temp_reg2, asmjit::x86::xmm0);
    Cb.push(temp_reg2);
    Cb.crc32(temp_reg, temp_reg2);
    Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
    Cb.movq(temp_reg2, asmjit::x86::xmm0);
    Cb.push(temp_reg2);
    Cb.crc32(temp_reg, temp_reg2);

  } else if constexpr (std::is_same<asmjit::x86::Ymm, VectorRegT>::value &&
                       std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    Cb.movq(temp_reg2, asmjit::x86::xmm0);
    Cb.movq(asmjit::x86::Mm(7), temp_reg2);
    Cb.crc32(temp_reg, temp_reg2);
    Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
    Cb.movq(temp_reg2, asmjit::x86::xmm0);
    Cb.movq(asmjit::x86::Mm(6), temp_reg2);
    Cb.crc32(temp_reg, temp_reg2);

    Cb.vextractf128(asmjit::x86::xmm0, asmjit::x86::ymm0, asmjit::Imm(1));

    Cb.movq(temp_reg2, asmjit::x86::xmm0);
    Cb.movq(asmjit::x86::Mm(5), temp_reg2);
    Cb.crc32(temp_reg, temp_reg2);
    Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
    Cb.movq(temp_reg2, asmjit::x86::xmm0);
    Cb.movq(asmjit::x86::Mm(4), temp_reg2);
    Cb.crc32(temp_reg, temp_reg2);
  } else if constexpr (std::is_same<asmjit::x86::Zmm, VectorRegT>::value &&
                       std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    // We use vector registers zmm31 for our backup
    Cb.vmovapd(asmjit::x86::zmm31, asmjit::x86::zmm0);
    registerCount--;
  }

  // Calculate the hash of the remaining VectorReg
  // use VectorReg(0) as a temporary place to unpack values
  for (int i = 1; i < registerCount; i++) {
    if constexpr (std::is_same<asmjit::x86::Xmm, VectorRegT>::value) {
      Cb.vmovapd(asmjit::x86::xmm0, asmjit::x86::Xmm(i));

      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
    } else if constexpr (std::is_same<asmjit::x86::Ymm, VectorRegT>::value) {
      Cb.vmovapd(asmjit::x86::ymm0, asmjit::x86::Ymm(i));

      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);

      Cb.vextractf128(asmjit::x86::xmm0, asmjit::x86::ymm0, asmjit::Imm(1));

      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
    } else if constexpr (std::is_same<asmjit::x86::Zmm, VectorRegT>::value) {
      Cb.vmovapd(asmjit::x86::ymm0, asmjit::x86::Ymm(i));

      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);

      Cb.vextractf128(asmjit::x86::xmm0, asmjit::x86::ymm0, asmjit::Imm(1));

      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);

      Cb.vextractf32x4(asmjit::x86::xmm0, asmjit::x86::Zmm(i), asmjit::Imm(2));

      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);

      Cb.vextractf32x4(asmjit::x86::xmm0, asmjit::x86::Zmm(i), asmjit::Imm(3));

      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(temp_reg2, asmjit::x86::xmm0);
      Cb.crc32(temp_reg, temp_reg2);
    }
  }

  // Restore VectorReg(0) from backup
  if constexpr (std::is_same<asmjit::x86::Xmm, VectorRegT>::value) {
    Cb.pop(temp_reg2);
    Cb.movq(asmjit::x86::xmm0, temp_reg2);
    Cb.movlhps(asmjit::x86::xmm0, asmjit::x86::xmm0);
    Cb.pop(temp_reg2);
    Cb.pinsrw(asmjit::x86::xmm0, temp_reg2.r32(), asmjit::Imm(0));
    Cb.shr(temp_reg2, asmjit::Imm(32));
    Cb.movd(temp_reg2.r32(), asmjit::x86::Mm(7));
    Cb.pinsrw(asmjit::x86::xmm0, temp_reg2.r32(), asmjit::Imm(1));
  } else if constexpr (std::is_same<asmjit::x86::Ymm, VectorRegT>::value &&
                       std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    Cb.movq(temp_reg2, asmjit::x86::Mm(5));
    Cb.movq(asmjit::x86::xmm0, temp_reg2);
    Cb.movq(temp_reg2, asmjit::x86::Mm(4));
    Cb.pinsrq(asmjit::x86::xmm0, temp_reg2, asmjit::Imm(1));

    Cb.vinsertf128(asmjit::x86::ymm0, asmjit::x86::ymm0, asmjit::x86::xmm0, asmjit::Imm(1));

    Cb.movq(temp_reg2, asmjit::x86::Mm(7));
    Cb.movq(asmjit::x86::xmm0, temp_reg2);
    Cb.movq(temp_reg2, asmjit::x86::Mm(6));
    Cb.pinsrq(asmjit::x86::xmm0, temp_reg2, asmjit::Imm(1));
  } else if constexpr (std::is_same<asmjit::x86::Zmm, VectorRegT>::value &&
                       std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    // We use vector registers zmm31 for our backup
    Cb.vmovapd(asmjit::x86::zmm0, asmjit::x86::zmm31);
  }

  // before starting the communication, backup r8, r9, rax, rbx, rcx and rdx
  if constexpr (std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    Cb.movq(asmjit::x86::Mm(7), asmjit::x86::rax);
    Cb.movq(asmjit::x86::Mm(6), asmjit::x86::rbx);
    Cb.movq(asmjit::x86::Mm(5), asmjit::x86::rcx);
    Cb.movq(asmjit::x86::Mm(4), asmjit::x86::rdx);
    Cb.movq(asmjit::x86::Mm(3), asmjit::x86::r8);
    Cb.movq(asmjit::x86::Mm(2), asmjit::x86::r9);
  } else {
    Cb.push(asmjit::x86::rax);
    Cb.push(asmjit::x86::rbx);
    Cb.push(asmjit::x86::rcx);
    Cb.push(asmjit::x86::rdx);
    Cb.push(asmjit::x86::r8);
    Cb.push(asmjit::x86::r9);
  }

  // do the actual communication
  // temp_reg contains our hash

  // save the pointer_reg. it might be any of r8, r9, rax, rbx, rcx or rdx
  Cb.mov(temp_reg2, pointer_reg);

  // Don't touch me!
  // This sychronization and communication works even if the threads run at
  // different (changing) speed, with just one "lock cmpxchg16b" Brought to you
  // by a few hours of headache for two people.
  auto communication = [&](auto offset) {
    // communication
    Cb.mov(asmjit::x86::r8, asmjit::x86::ptr_64(temp_reg2, offset));

    // temp data
    Cb.mov(asmjit::x86::r9, temp_reg2);
    Cb.add(asmjit::x86::r9, asmjit::Imm(offset + 8));

    Cb.mov(asmjit::x86::rdx, asmjit::x86::ptr_64(asmjit::x86::r9, 0));
    Cb.mov(asmjit::x86::rax, asmjit::x86::ptr_64(asmjit::x86::r9, 8));

    auto L0 = Cb.newLabel();
    Cb.bind(L0);

    Cb.lock();
    Cb.cmpxchg16b(asmjit::x86::ptr(asmjit::x86::r8));

    auto L1 = Cb.newLabel();
    Cb.jnz(L1);

    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 0), asmjit::x86::rcx);
    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 8), asmjit::x86::rbx);
    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 16), asmjit::Imm(0));
    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 24), asmjit::Imm(0));

    Cb.mov(asmjit::x86::rax, asmjit::Imm(2));

    auto L6 = Cb.newLabel();
    Cb.jmp(L6);

    Cb.bind(L1);

    Cb.cmp(asmjit::x86::rcx, asmjit::x86::rdx);

    auto L2 = Cb.newLabel();
    Cb.jle(L2);

    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 0), asmjit::x86::rcx);
    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 8), asmjit::x86::rbx);

    Cb.jmp(L0);

    Cb.bind(L2);

    auto L3 = Cb.newLabel();

    Cb.cmp(asmjit::x86::ptr_64(asmjit::x86::r9, 16), asmjit::Imm(0));
    Cb.jne(L3);
    Cb.cmp(asmjit::x86::ptr_64(asmjit::x86::r9, 24), asmjit::Imm(0));
    Cb.jne(L3);

    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 16), asmjit::x86::rdx);
    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 24), asmjit::x86::rax);

    Cb.bind(L3);

    Cb.cmp(asmjit::x86::rcx, asmjit::x86::ptr_64(asmjit::x86::r9, 16));
    Cb.mov(asmjit::x86::rax, asmjit::Imm(4));
    Cb.jne(L6);

    Cb.cmp(asmjit::x86::rbx, asmjit::x86::ptr_64(asmjit::x86::r9, 24));
    auto L4 = Cb.newLabel();
    Cb.jne(L4);

    Cb.mov(asmjit::x86::rax, asmjit::Imm(0));

    auto L5 = Cb.newLabel();
    Cb.jmp(L5);

    Cb.bind(L4);

    Cb.mov(asmjit::x86::rax, asmjit::Imm(1));

    Cb.bind(L5);

    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 16), asmjit::Imm(0));
    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 24), asmjit::Imm(0));

    Cb.bind(L6);

    // if check failed
    Cb.cmp(asmjit::x86::rax, asmjit::Imm(1));
    auto L7 = Cb.newLabel();
    Cb.jne(L7);

    // write the error flag
    Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, 32), asmjit::Imm(1));

    // stop the execution after some time
    Cb.mov(asmjit::x86::ptr_64(addrHigh_reg), asmjit::Imm(LOAD_STOP));
    Cb.mfence();

    Cb.bind(L7);

    auto L9 = Cb.newLabel();
    Cb.jmp(L9);
  };

  // left communication
  // move hash
  Cb.mov(asmjit::x86::rbx, temp_reg);
  // move iterations counter
  if constexpr (std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    Cb.movq(asmjit::x86::rcx, IterReg);
  } else {
    Cb.mov(asmjit::x86::rcx, IterReg);
  }

  communication(-128);

  // right communication
  // move hash
  Cb.mov(asmjit::x86::rbx, temp_reg);
  // move iterations counter
  if constexpr (std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    Cb.movq(asmjit::x86::rcx, IterReg);
  } else {
    Cb.mov(asmjit::x86::rcx, IterReg);
  }

  communication(-64);

  // restore r8, r9, rax, rbx, rcx and rdx
  if constexpr (std::is_same<asmjit::x86::Mm, IterRegT>::value) {
    Cb.movq(asmjit::x86::rax, asmjit::x86::Mm(7));
    Cb.movq(asmjit::x86::rbx, asmjit::x86::Mm(6));
    Cb.movq(asmjit::x86::rcx, asmjit::x86::Mm(5));
    Cb.movq(asmjit::x86::rdx, asmjit::x86::Mm(4));
    Cb.movq(asmjit::x86::r8, asmjit::x86::Mm(3));
    Cb.movq(asmjit::x86::r9, asmjit::x86::Mm(2));
  } else {
    Cb.pop(asmjit::x86::r9);
    Cb.pop(asmjit::x86::r8);
    Cb.pop(asmjit::x86::rdx);
    Cb.pop(asmjit::x86::rcx);
    Cb.pop(asmjit::x86::rbx);
    Cb.pop(asmjit::x86::rax);
  }

  Cb.bind(SkipErrorDetection);
}

template void X86Payload::emitErrorDetectionCode<asmjit::x86::Gpq, asmjit::x86::Xmm>(
    asmjit::x86::Builder& cb, asmjit::x86::Gpq iter_reg, asmjit::x86::Gpq addrHigh_reg, asmjit::x86::Gpq pointer_reg,
    asmjit::x86::Gpq temp_reg, asmjit::x86::Gpq temp_reg2);
template void X86Payload::emitErrorDetectionCode<asmjit::x86::Gpq, asmjit::x86::Ymm>(
    asmjit::x86::Builder& cb, asmjit::x86::Gpq iter_reg, asmjit::x86::Gpq addrHigh_reg, asmjit::x86::Gpq pointer_reg,
    asmjit::x86::Gpq temp_reg, asmjit::x86::Gpq temp_reg2);

template void X86Payload::emitErrorDetectionCode<asmjit::x86::Mm, asmjit::x86::Ymm>(
    asmjit::x86::Builder& cb, asmjit::x86::Mm iter_reg, asmjit::x86::Gpq addrHigh_reg, asmjit::x86::Gpq pointer_reg,
    asmjit::x86::Gpq temp_reg, asmjit::x86::Gpq temp_reg2);
template void X86Payload::emitErrorDetectionCode<asmjit::x86::Mm, asmjit::x86::Zmm>(
    asmjit::x86::Builder& cb, asmjit::x86::Mm iter_reg, asmjit::x86::Gpq addrHigh_reg, asmjit::x86::Gpq pointer_reg,
    asmjit::x86::Gpq temp_reg, asmjit::x86::Gpq temp_reg2);
