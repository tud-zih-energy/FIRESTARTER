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

#include "firestarter/Environment/X86/Payload/X86Payload.hpp"
#include "firestarter/Constants.hpp"
#include "firestarter/WindowsCompat.hpp"

#include <cassert>
#include <chrono>
#include <thread>

namespace firestarter::environment::x86::payload {

void X86Payload::lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) const {
  auto Nap = Period / 100;

  if constexpr (firestarter::OptionalFeatures.IsMsc) {
    std::array<int, 4> Cpuid{};
    _mm_mfence();
    __cpuid(Cpuid.data(), 0);
  } else {
    __asm__ __volatile__("mfence;"
                         "cpuid;" ::
                             : "eax", "ebx", "ecx", "edx");
  }

  // while signal low load
  while (LoadVar == LoadThreadWorkType::LoadLow) {
    if constexpr (firestarter::OptionalFeatures.IsMsc) {
      std::array<int, 4> Cpuid{};
      _mm_mfence();
      __cpuid(Cpuid.data(), 0);
    } else {
      __asm__ __volatile__("mfence;"
                           "cpuid;" ::
                               : "eax", "ebx", "ecx", "edx");
    }
    std::this_thread::sleep_for(Nap);
    if constexpr (firestarter::OptionalFeatures.IsMsc) {
      std::array<int, 4> Cpuid{};
      _mm_mfence();
      __cpuid(Cpuid.data(), 0);
    } else {
      __asm__ __volatile__("mfence;"
                           "cpuid;" ::
                               : "eax", "ebx", "ecx", "edx");
    }
  }
}

void X86Payload::initMemory(double* MemoryAddr, uint64_t BufferSize, double FirstValue, double LastValue) {
  uint64_t I = 0;

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (; I < InitBlocksize; I++) {
    MemoryAddr[I] = 0.25 + static_cast<double>(I) * 8.0 * FirstValue;
  }
  for (; I <= BufferSize - InitBlocksize; I += InitBlocksize) {
    std::memcpy(MemoryAddr + I, MemoryAddr + I - InitBlocksize, sizeof(uint64_t) * InitBlocksize);
  }
  for (; I < BufferSize; I++) {
    MemoryAddr[I] = 0.25 + static_cast<double>(I) * 8.0 * LastValue;
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

auto X86Payload::getAvailableInstructions() const -> std::list<std::string> {
  std::list<std::string> Instructions;

  transform(InstructionFlops.begin(), InstructionFlops.end(), back_inserter(Instructions),
            [](const auto& Item) { return Item.first; });

  return Instructions;
}

}; // namespace firestarter::environment::x86::payload