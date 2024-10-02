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

#include <cassert>
#include <chrono>
#include <thread>

#ifdef _MSC_VER
#include <array>
#include <intrin.h>
#endif

#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::payload {

void X86Payload::lowLoadFunction(volatile uint64_t* AddrHigh, uint64_t Period) {
  int Nap = Period / 100;

#ifndef _MSC_VER
  __asm__ __volatile__("mfence;"
                       "cpuid;" ::
                           : "eax", "ebx", "ecx", "edx");
#else
  std::array<int, 4> Cpuid;
  _mm_mfence();
  __cpuid(Cpuid.data(), 0);
#endif

  // while signal low load
  while (*AddrHigh == LOAD_LOW) {
#ifndef _MSC_VER
    __asm__ __volatile__("mfence;"
                         "cpuid;" ::
                             : "eax", "ebx", "ecx", "edx");
#else
    _mm_mfence();
    __cpuid(Cpuid.data(), 0);
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(Nap));
#ifndef _MSC_VER
    __asm__ __volatile__("mfence;"
                         "cpuid;" ::
                             : "eax", "ebx", "ecx", "edx");
#else
    _mm_mfence();
    __cpuid(Cpuid.data(), 0);
#endif
  }
}

void X86Payload::init(uint64_t* MemoryAddr, uint64_t BufferSize, double FirstValue, double LastValue) {
  uint64_t i = 0;

  for (; i < INIT_BLOCKSIZE; i++) {
    reinterpret_cast<double*>(MemoryAddr)[i] = 0.25 + static_cast<double>(i) * 8.0 * FirstValue;
  }
  for (; i <= BufferSize - INIT_BLOCKSIZE; i += INIT_BLOCKSIZE) {
    std::memcpy(MemoryAddr + i, MemoryAddr + i - INIT_BLOCKSIZE, sizeof(uint64_t) * INIT_BLOCKSIZE);
  }
  for (; i < BufferSize; i++) {
    reinterpret_cast<double*>(MemoryAddr)[i] = 0.25 + static_cast<double>(i) * 8.0 * LastValue;
  }
}

auto X86Payload::highLoadFunction(uint64_t* AddrMem, volatile uint64_t* AddrHigh, uint64_t Iterations) -> uint64_t {
  return this->LoadFunction(AddrMem, AddrHigh, Iterations);
}

}; // namespace firestarter::environment::x86::payload