/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2022-2023 TU Dresden, Center for Information Services and High
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

#pragma once

#include "firestarter/Constants.hpp"
#include <array>

namespace firestarter {

/* DO NOT CHANGE! the asm load-loop tests if it should dump the current register
 * content */
// NOLINTBEGIN(performance-enum-size)
/// This struct defines the variable the is used to control when the registers should be dumped.
enum class DumpVariable : EightBytesType {
  /// Start saving register to memory
  Start = 0,
  /// When done when change it to the Wait state. There we do nothing.
  Wait = 1
};
// NOLINTEND(performance-enum-size)

// The maximal number of SIMD registers. This is currently 32 for zmm registers.
constexpr const auto RegisterMaxNum = 32;
/// The maximal number of doubles in SIMD registers. This is currently 8 for zmm registers.
constexpr const auto RegisterMaxSize = 8;
/// The maximum number of doubles in SIMD registers multiplied with the maximum number of vector registers.
constexpr const auto MaxNumberOfDoublesInVectorRegisters = RegisterMaxNum * RegisterMaxSize;

/// This struct is used to do the communication between the high-load loop and the part of the program that saves the
/// dumped registers to a file.
struct DumpRegisterStruct {
  /// This array will contain the dumped registers. It has the size of 32 Cachelines. (8B doubles * 8 double in a
  /// register * 32 registers)
  std::array<double, MaxNumberOfDoublesInVectorRegisters> RegisterValues;
  /// Pad the DumpVar to use a whole cacheline
  std::array<EightBytesType, 7> Padding;
  /// The variable that controls the execution of the dump register code in the high-load routine.
  volatile DumpVariable DumpVar;
};

} // namespace firestarter
