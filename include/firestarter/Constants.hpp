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

enum class LoadThreadState : EightBytesType { ThreadWait = 1, ThreadWork = 2, ThreadInit = 3, ThreadSwitch = 4 };

enum class LoadThreadWorkType : EightBytesType {
  /* DO NOT CHANGE! the asm load-loop tests if load-variable is == 0 */
  LoadLow = 0,
  /* DO NOT CHANGE! the asm load-loop continues until the load-variable is != 1 */
  LoadHigh = 1,
  LoadStop = 2,
  LoadSwitch = 4
};
// NOLINTEND(performance-enum-size)

/// This struct holds infomation about enabled or disabled compile time features for FIRESTARTER.
struct FirestarterOptionalFeatures {
  /// Do we have a build that enabled optimization?
  bool OptimizationEnabled = false;
  /// Do we have a build that enabled CUDA or HIP?
  bool CudaEnabled = false;
  /// Do we have a build that enabled OneAPU?
  bool OneAPIEnabled = false;
  /// Is error detection enabled?
  bool ErrorDetectionEnabled = false;
  /// Are debug features enabled?
  bool DebugFeatureEnabled = false;
  /// Is dumping registers enabled?
  bool DumpRegisterEnabled = false;
  /// Is the current build for X86?
  bool IsX86 = false;
  /// Is the current build for Windows?
  bool IsWin32 = false;
  /// Is the current build built with Windows MSC?
  bool IsMsc = false;

  /// Is one of the GPU features enabled?
  [[nodiscard]] constexpr auto gpuEnabled() const -> bool { return CudaEnabled || OneAPIEnabled; }
};

static constexpr const FirestarterOptionalFeatures OptionalFeatures {
#if defined(linux) || defined(__linux__)
  .OptimizationEnabled = true,
#endif
#if defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP)
  .CudaEnabled = true,
#endif
#ifdef FIRESTARTER_BUILD_ONEAPI
  .OneAPIEnabled = true,
#endif
  .ErrorDetectionEnabled = true,
#ifdef FIRESTARTER_DEBUG_FEATURES
  .DebugFeatureEnabled = true, .DumpRegisterEnabled = true,
#endif
#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
  .IsX86 = true,
#else
#error "FIRESTARTER is not implemented for this ISA"
#endif
#ifdef _WIN32
  .IsWin32 = true,
#endif
#ifdef _MSC_VER
  .IsMsc = true,
#endif
};

} // namespace firestarter