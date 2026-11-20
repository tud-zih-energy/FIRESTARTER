/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
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

#include "firestarter/AArch64/Payload/AArch64Payload.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iterator>
#include <thread>

namespace firestarter::aarch64::payload {

void AArch64Payload::lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) const {
  auto Nap = Period / 100;

  __asm__ __volatile__("dmb sy" : : : "memory");

  // while signal low load
  while (LoadVar == LoadThreadWorkType::LoadLow) {
    __asm__ __volatile__("dmb sy" : : : "memory");
    std::this_thread::sleep_for(std::chrono::microseconds(Nap));
    __asm__ __volatile__("dmb sy" : : : "memory");
  }
}

void AArch64Payload::initMemory(double* MemoryAddr, uint64_t BufferSize, double FirstValue, double LastValue) {
  uint64_t I = 0;

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (; I < AArch64InitBlocksize; I++) {
    MemoryAddr[I] = 0.25 + static_cast<double>(I) * 8.0 * FirstValue;
  }
  for (; I <= BufferSize - AArch64InitBlocksize; I += AArch64InitBlocksize) {
    std::memcpy(MemoryAddr + I, MemoryAddr + I - AArch64InitBlocksize, sizeof(uint64_t) * AArch64InitBlocksize);
  }
  for (; I < BufferSize; I++) {
    MemoryAddr[I] = 0.25 + static_cast<double>(I) * 8.0 * LastValue;
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

auto AArch64Payload::getAvailableInstructions() const -> std::list<std::string> {
  std::list<std::string> Instructions;

  std::transform(InstructionFlops.begin(), InstructionFlops.end(), std::back_inserter(Instructions),
                 [](const auto& Item) { return Item.first; });

  return Instructions;
}

} // namespace firestarter::aarch64::payload
