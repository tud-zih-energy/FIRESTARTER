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

namespace firestarter {

/* DO NOT CHANGE! the asm load-loop tests if it should dump the current register
 * content */
// NOLINTBEGIN(performance-enum-size)
// Define the variable with the size of a cache line
enum class DumpVariable : EightBytesType { Start = 0, Wait = 1 };
// NOLINTEND(performance-enum-size)

// The maximal number of SIMD registers. This is currently 32 for zmm registers.
constexpr const auto RegisterMaxNum = 32;
/// The maximal number of doubles in SIMD registers. This is currently 8 for zmm registers.
constexpr const auto RegisterMaxSize = 8;

// REGISTER_MAX_NUM cachelines
struct DumpRegisterStruct {
  volatile double RegisterValues[RegisterMaxNum * RegisterMaxSize];
  // pad to use a whole cacheline
  volatile EightBytesType Padding[7];
  volatile DumpVariable DumpVar;
};

#undef REGISTER_MAX_NUM

} // namespace firestarter
