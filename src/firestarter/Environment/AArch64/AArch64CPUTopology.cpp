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

#include <firestarter/Environment/AArch64/AArch64CPUTopology.hpp>
#include <firestarter/Logging/Log.hpp>

#include <ctime>

#ifdef _MSC_VER
#include <array>
#include <intrin.h>

#pragma intrinsic(__rdtsc)
#endif

using namespace firestarter::environment::aarch64;

AArch64CPUTopology::AArch64CPUTopology()
    : CPUTopology("AArch64"), cpuInfo(asmjit::CpuInfo::host()),
      _vendor(this->cpuInfo.vendor()) {

}

// measures clockrate using the Time-Stamp-Counter
// only constant TSCs will be used (i.e. power management indepent TSCs)
// save frequency in highest P-State or use generic fallback if no invarient TSC
// is available
unsigned long long AArch64CPUTopology::clockrate() const {
  return 0;
}

unsigned long long AArch64CPUTopology::timestamp() const {
  return 0;
}
