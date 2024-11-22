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

#include <cstdint>

namespace firestarter {

/// This struct is used for the error detection feature. The error detection works between two threads. The current one
/// and one on the left. Analogous for the thread on the right. We hash the contents of the vector registers and compare
/// them with the current iteration counter aginst the other threads.
struct ErrorDetectionStruct {
  struct OneSide {
    /// The pointer to 16B of communication between the two threads which is used with lock cmpxchg16b
    uint64_t* Communication;
    /// The local variables that are used for the error detection algorithm
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    uint64_t Locals[4];
    /// If this variable is not 0, an error occured in the comparison with the other thread.
    uint64_t Error;
    /// Padding to fill up a cache line.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    uint64_t Padding[2];
  };

  /// The data that is used for the error detection algorithm between the current and the thread left to it.
  OneSide Left;
  /// The data that is used for the error detection algorithm between the current and the thread right to it.
  OneSide Right;
};

} // namespace firestarter