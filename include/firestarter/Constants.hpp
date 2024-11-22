/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2021 TU Dresden, Center for Information Services and High
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

using EightBytesType = uint64_t;

// We want enum to have the size of 8B. Disable the warnings for bigger enum size than needed.
// NOLINTBEGIN(performance-enum-size)

/// This enum describes the state of the load workers.
enum class LoadThreadState : EightBytesType {
  /// Idle
  ThreadWait = 1,
  /// Work loop (both low and high load)
  ThreadWork = 2,
  /// Init the thread
  ThreadInit = 3,
  /// Tell the thread to recompile the payload and reinitialize the data.
  ThreadSwitch = 4
};

/// This enum describes the Load that should be applied by firestarter.
enum class LoadThreadWorkType : EightBytesType {
  /* DO NOT CHANGE! the asm load-loop tests if load-variable is == 0 */
  /// Apply low load
  LoadLow = 0,
  /* DO NOT CHANGE! the asm load-loop continues until the load-variable is != 1 */
  /// Apply hugh load
  LoadHigh = 1,
  /// Exit the load loop and stop the execution of firestarter.
  LoadStop = 2,
  /// Exit the load loop.
  LoadSwitch = 4
};
// NOLINTEND(performance-enum-size)

/// This struct holds infomation about enabled or disabled compile time features for FIRESTARTER.
struct FirestarterOptionalFeatures {
  /// Do we have a build that enabled optimization?
  bool OptimizationEnabled;
  /// Do we have a build that enabled CUDA or HIP?
  bool CudaEnabled;
  /// Do we have a build that enabled OneAPU?
  bool OneAPIEnabled;
  /// Is error detection enabled?
  bool ErrorDetectionEnabled;
  /// Are debug features enabled?
  bool DebugFeatureEnabled;
  /// Is dumping registers enabled?
  bool DumpRegisterEnabled;
  /// Is the current build for X86?
  bool IsX86;
  /// Is the current build for Windows?
  bool IsWin32;
  /// Is the current build built with Windows MSC?
  bool IsMsc;

  /// Is one of the GPU features enabled?
  [[nodiscard]] constexpr auto gpuEnabled() const -> bool { return CudaEnabled || OneAPIEnabled; }
};

// MSC only supports designated initializers from C++20
static constexpr const FirestarterOptionalFeatures OptionalFeatures {
#if defined(linux) || defined(__linux__)
  /*OptimizationEnabled=*/true,
#else
  /*OptimizationEnabled=*/false,
#endif

#if defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP)
      /*CudaEnabled=*/true,
#else
      /*CudaEnabled=*/false,
#endif

#ifdef FIRESTARTER_BUILD_ONEAPI
      /*OneAPIEnabled=*/true,
#else
      /*OneAPIEnabled=*/false,
#endif

      /*ErrorDetectionEnabled=*/true,

#ifdef FIRESTARTER_DEBUG_FEATURES
      /*DebugFeatureEnabled=*/true, /*DumpRegisterEnabled =*/true,
#else
      /*DebugFeatureEnabled=*/false, /*DumpRegisterEnabled =*/false,
#endif

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
      /*IsX86=*/true,
#else
#error "FIRESTARTER is not implemented for this ISA"
#endif

#ifdef _WIN32
      /*IsWin32=*/true,
#else
      /*IsWin32=*/false,
#endif

#ifdef _MSC_VER
      /*IsMsc=*/true,
#else
      /*IsMsc=*/false,
#endif
};

} // namespace firestarter