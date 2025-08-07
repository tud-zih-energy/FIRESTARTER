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

#pragma once

#include "firestarter/AlignedAlloc.hpp"
#include "firestarter/DumpRegisterStruct.hpp"
#include "firestarter/ErrorDetectionStruct.hpp"

#include <memory>

namespace firestarter {

/// This struct is used to allocate the memory for the high-load routine.
struct LoadWorkerMemory {
private:
  LoadWorkerMemory() = default;
  ~LoadWorkerMemory() = default;

  /// Function to deallocate the memory for this struct to be used with unique_ptr.
  /// \arg Ptr The pointer to the memory
  static void deallocate(void* Ptr) {
    static_cast<LoadWorkerMemory*>(Ptr)->~LoadWorkerMemory();
    AlignedAlloc::free(Ptr);
  }

public:
  using UniquePtr = std::unique_ptr<LoadWorkerMemory, void (*)(void*)>;

  /// The extra variables that are before the memory used for the calculation in the high-load routine. They are used
  /// for optional FIRESTARTER features where further communication between the high-load routine is needed e.g., for
  /// error detection or dumping registers.
  struct ExtraLoadWorkerVariables {
    /// The data for the dump registers functionality.
    DumpRegisterStruct Drs;
    /// The data for the error detections functionality.
    ErrorDetectionStruct Eds;
    // Define struct that is used as config and loaded through ldtilecfg()
    // 64 Byte aligned
    struct TileConfig {
      uint8_t palette_id;
      uint8_t start_row;
      uint8_t reserved_0[14];
      uint16_t colsb[16];
      uint8_t rows[16];
    } Tc;
    
    struct AMXMemory {
      uint16_t src1[1024];
      uint16_t src2[1024];
      uint16_t src3[1024];
    } AMXmem;
    
  } ExtraVars;

  /// A placeholder to extract the address of the memory region with dynamic size which is used for the calculation in
  /// the high-load routine. Do not write or read to this type directly.
  EightBytesType DoNotUseAddrMem;

  /// This padding makes shure that we are aligned to a cache line. The allocated memory will most probably reach beyond
  /// this array.
  std::array<EightBytesType, 7> DoNotUsePadding;

  /// Get the pointer to the start of the memory use for computations.
  /// \returns the pointer to the memory.
  [[nodiscard]] auto getMemoryAddress() -> auto {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<double*>(&DoNotUseAddrMem);
  }

  /// Get the offset to the memory which is used by the high-load functions
  /// \returns the offset to the memory
  [[nodiscard]] constexpr static auto getMemoryOffset() -> auto { return offsetof(LoadWorkerMemory, DoNotUseAddrMem); }

  /// Allocate the memory for the high-load thread on 64B cache line boundaries and return a unique_ptr.
  /// \arg Bytes The number of bytes allocated for the array whoose start address is returned by the getMemoryAddress
  /// function.
  /// \returns A unique_ptr to the memory for the high-load thread.
  [[nodiscard]] static auto allocate(const std::size_t Bytes) -> UniquePtr {
    // Allocate the memory for the ExtraLoadWorkerVariables (which are 64B aligned) and the data for the high-load
    // routine which may not be 64B aligned.
    static_assert(sizeof(ExtraLoadWorkerVariables) % 64 == 0,
                  "ExtraLoadWorkerVariables is not a multiple of 64B i.e., multiple cachelines.");
    auto* Ptr = AlignedAlloc::malloc(Bytes + sizeof(ExtraLoadWorkerVariables));
    return {static_cast<LoadWorkerMemory*>(Ptr), deallocate};
  }
};

} // namespace firestarter
