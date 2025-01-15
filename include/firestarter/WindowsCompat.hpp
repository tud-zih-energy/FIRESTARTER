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

#ifdef _MSC_VER
#include <intrin.h>
#else

/// Define the _mm_mfence and __cpuid function when we are not using MSC to enable the use of if constexpr instead of
/// ifdefs.
// NOLINTBEGIN(readability-identifier-naming,cert-dcl37-c,cert-dcl37-cpp,cert-dcl51-cpp,bugprone-reserved-identifier)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#if defined(__clang__)
#include <emmintrin.h>
#elif not(defined(__MINGW32__) || defined(__MINGW64__))
void _mm_mfence() noexcept;
#else
#include <intrin.h>
#endif

#if not(defined(__INTEL_LLVM_COMPILER))
void __cpuid(int* /*unused*/, int /*unused*/) noexcept;
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
// NOLINTEND(readability-identifier-naming,cert-dcl37-c,cert-dcl37-cpp,cert-dcl51-cpp,bugprone-reserved-identifier)

#endif

#ifdef _WIN32
// SIGALRM is not available on Windows
#define SIGALRM 0

#include <direct.h>
static inline auto get_current_dir_name() -> char* { return _getcwd(nullptr, 0); }
#elif defined(__APPLE__)
#include <unistd.h>
static inline auto get_current_dir_name() -> char* { return getcwd(nullptr, 0); }
#else
#include <unistd.h>
#endif

// correct include for gethostname
#ifdef _MSC_VER
#include <winsock.h>
#else
// NOLINTBEGIN(readability-duplicate-include)
#include <unistd.h>
// NOLINTEND(readability-duplicate-include)
#endif

// Make references in header files to pthread_t compatible to MSC. This will not make them functionally work.
// We will be able to remove this hack once we transition from using pthread to std::thread
#ifdef _MSC_VER
struct Placeholder {};
using pthread_t = Placeholder;
#else
extern "C" {
#include <pthread.h>
}
#endif

// Disable __asm__ __volatile__ in MSC
// Static assert wont work, since if constexpr doesn't seem to work correctly
#ifdef _MSC_VER
#define __volatile__(X, ...)                                                                                           \
  assert(false && "Attempted to use code path that uses the incorrect inline assembly macros for MSC.")
#define __asm__
#endif