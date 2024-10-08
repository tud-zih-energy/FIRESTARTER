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

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace firestarter {

struct AlignedAlloc {
private:
  /// Round the size to the nearest multiple of the aligment
  /// \arg Size The number to be rounded up.
  /// \arg Alignment The number to whoose multiple to be round up to.
  /// \returns Size rounded up to the nearest multiple of the Alignment
  static auto padSize(const std::size_t Size, const std::size_t Alignment) -> std::size_t {
    return Alignment * static_cast<int>(std::ceil(static_cast<double>(Size) / static_cast<double>(Alignment)));
  };

public:
  /// Allocate memory with a given alignment. The size will automatically increased to the nearest multiple of the
  /// alignment.
  /// \arg Size The minimum required memory.
  /// \arg Alignment describes to which boundary the memory should be aligned. The default is 64B which will account to
  /// the size of a cache line on most systems.
  /// \returns The pointer to the allocated memory.
  static auto malloc(const std::size_t Size, const std::size_t Alignment = 64) -> void* {
    // NOLINTBEGIN(cppcoreguidelines-owning-memory)
#if defined(__APPLE__)
    return aligned_alloc(Alignment, padSize(Size, Alignment));
#elif defined(__MINGW64__)
    return _mm_malloc(padSize(Size, Alignment), Alignment);
#elif defined(_MSC_VER)
    return _aligned_malloc(padSize(Size, Alignment), Alignment);
#else
    return aligned_alloc(Alignment, padSize(Size, Alignment));
#endif
    // NOLINTEND(cppcoreguidelines-owning-memory)
  };

  /// Deallocate memory which has been allocated by the AlignedAlloc::malloc function.
  /// \arg Ptr The pointer to the allocated memory.
  static void free(void* Ptr) {
    // NOLINTBEGIN(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#if defined(__APPLE__)
    ::free(Ptr);
#elif defined(__MINGW64__)
    _mm_free(Ptr);
#elif defined(_MSC_VER)
    _aligned_free(Ptr);
#else
    std::free(Ptr);
#endif
    // NOLINTEND(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
  };
};

} // namespace firestarter
