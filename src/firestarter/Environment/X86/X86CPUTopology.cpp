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

#include <firestarter/Environment/X86/X86CPUTopology.hpp>
#include <firestarter/Logging/Log.hpp>

#include <ctime>

#ifdef _MSC_VER
#include <array>
#include <intrin.h>

#pragma intrinsic(__rdtsc)
#endif

namespace firestarter::environment::x86 {

X86CPUTopology::X86CPUTopology()
    : CPUTopology("x86_64")
    , CpuInfo(asmjit::CpuInfo::host()) {

  Vendor = CpuInfo.vendor();

  {
    std::stringstream Ss;
    Ss << "Family " << familyId() << ", Model " << modelId() << ", Stepping " << stepping();
    Model = Ss.str();
  }

  for (auto FeatureId = 0; FeatureId <= asmjit::CpuFeatures::X86::Id::kMaxValue; FeatureId++) {
    if (!CpuInfo.hasFeature(FeatureId)) {
      continue;
    }

    asmjit::String Sb;

    auto Error = asmjit::Formatter::formatFeature(Sb, CpuInfo.arch(), FeatureId);
    if (Error != asmjit::ErrorCode::kErrorOk) {
      log::warn() << "Formatting cpu features got asmjit error: " << Error;
    }

    FeatureList.emplace_back(Sb.data());
  }

  uint64_t Rax = 0;
  uint64_t Rbx = 0;
  uint64_t Rcx = 0;
  uint64_t Rdx = 0;

  // check if we have rdtsc
  cpuid(&Rax, &Rbx, &Rcx, &Rdx);
  if (Rax >= 1) {
    Rax = 1;
    cpuid(&Rax, &Rbx, &Rcx, &Rdx);
    HasRdtsc = (Rdx & (1 << 4)) != 0;
  }

  // check if we have invarant rdtsc
  if (hasRdtsc()) {
    Rax = 0, Rbx = 0, Rcx = 0, Rdx = 0;

    HasInvariantRdtsc = true;

    /* TSCs are usable if CPU supports only one frequency in C0 (no
       speedstep/Cool'n'Quite)
       or if multiple frequencies are available and the constant/invariant TSC
       feature flag is set */

    if ("INTEL" == vendor()) {
      /*check if Powermanagement and invariant TSC are supported*/
      Rax = 1;
      cpuid(&Rax, &Rbx, &Rcx, &Rdx);
      /* no Frequency control */
      if ((!(Rdx & (1 << 22))) && (!(Rcx & (1 << 7)))) {
        HasInvariantRdtsc = true;
      } else {
        Rax = 0x80000000;
        cpuid(&Rax, &Rbx, &Rcx, &Rdx);
        if (Rax >= 0x80000007) {
          Rax = 0x80000007;
          cpuid(&Rax, &Rbx, &Rcx, &Rdx);
          /* invariant TSC */
          if (Rdx & (1 << 8)) {
            HasInvariantRdtsc = true;
          }
        }
      }
    }

    if ("AMD" == vendor()) {
      /*check if Powermanagement and invariant TSC are supported*/
      Rax = 0x80000000;
      cpuid(&Rax, &Rbx, &Rcx, &Rdx);
      if (Rax >= 0x80000007) {
        Rax = 0x80000007;
        cpuid(&Rax, &Rbx, &Rcx, &Rdx);

        /* no Frequency control */
        if ((!(Rdx & (1 << 7))) && (!(Rdx & (1 << 1)))) {
          HasInvariantRdtsc = true;
        }
        /* invariant TSC */
        if (Rdx & (1 << 8)) {
          HasInvariantRdtsc = true;
        }
      }
      /* assuming no frequency control if cpuid does not provide the extended
         function to test for it */
      else {
        HasInvariantRdtsc = true;
      }
    }
  }
}

// measures clockrate using the Time-Stamp-Counter
// only constant TSCs will be used (i.e. power management indepent TSCs)
// save frequency in highest P-State or use generic fallback if no invarient TSC
// is available
auto X86CPUTopology::clockrate() const -> uint64_t {
  using ClockT = std::chrono::high_resolution_clock;
  using TicksT = std::chrono::microseconds;

  uint64_t TimeDiff = 0;
  uint64_t Clockrate = 0;
  int NumMeasurements = 0;
  int MinMeasurements = 0;

  ClockT::time_point StartTime;
  ClockT::time_point EndTime;

#if not(defined(__APPLE__) || defined(_WIN32))
  auto Governor = scalingGovernor();
  if (Governor.empty()) {
    return CPUTopology::clockrate();
  }

  /* non invariant TSCs can be used if CPUs run at fixed frequency */
  if (!hasInvariantRdtsc() && Governor.compare("performance") && Governor.compare("powersave")) {
    return CPUTopology::clockrate();
  }

  MinMeasurements = 5;
#else
  MinMeasurements = 20;
#endif

  int I = 3;

  do {
    uint64_t End1Tsc = 0;
    uint64_t End2Tsc = 0;

    // start timestamp
    const uint64_t Start1Tsc = timestamp();
    StartTime = ClockT::now();
    const uint64_t Start2Tsc = timestamp();

    // waiting
    do {
      End1Tsc = timestamp();
    } while (End1Tsc < Start2Tsc + 1000000 * I); /* busy waiting */

    // end timestamp
    do {
      End1Tsc = timestamp();
      EndTime = ClockT::now();
      End2Tsc = timestamp();

      TimeDiff = std::chrono::duration_cast<TicksT>(EndTime - StartTime).count();
    } while (0 == TimeDiff);

    const uint64_t ClockLowerBound = (((End1Tsc - Start2Tsc) * 1000000) / (TimeDiff));
    const uint64_t ClockUpperBound = (((End2Tsc - Start1Tsc) * 1000000) / (TimeDiff));

    // if both values differ significantly, the measurement could have been
    // interrupted between 2 rdtsc's
    if ((static_cast<double>(ClockLowerBound) > ((static_cast<double>(ClockUpperBound)) * 0.999)) &&
        ((TimeDiff) > 2000)) {
      NumMeasurements++;
      const uint64_t Clock = (ClockLowerBound + ClockUpperBound) / 2;
      const bool ClockrateUpdateCondition = Clockrate == 0 ||
#ifndef _WIN32
                                            Clock < Clockrate;
#else
                                            Clock > Clockrate;
#endif
      if (ClockrateUpdateCondition) {
        Clockrate = Clock;
      }
    }
    I += 2;
  } while (((TimeDiff) < 10000) || (NumMeasurements < MinMeasurements));

  return Clockrate;
}

auto X86CPUTopology::timestamp() const -> uint64_t {
  if (!hasRdtsc()) {
    return 0;
  }

#ifndef _MSC_VER
  // NOLINTBEGIN(misc-const-correctness)
  uint64_t Rax = 0;
  uint64_t Rdx = 0;
  // NOLINTEND(misc-const-correctness)
  __asm__ __volatile__("rdtsc;" : "=a"(Rax), "=d"(Rdx));
  return (Rdx << 32) | (Rax & 0xffffffffULL);
#else
  return __rdtsc();
#endif
}

void X86CPUTopology::cpuid(uint64_t* Rax, uint64_t* Rbx, uint64_t* Rcx, uint64_t* Rdx) {
#ifndef _MSC_VER
  // NOLINTBEGIN(misc-const-correctness)
  uint64_t RaxOut = 0;
  uint64_t RbxOut = 0;
  uint64_t RcxOut = 0;
  uint64_t RdxOut = 0;
  // NOLINTEND(misc-const-correctness)
  __asm__ __volatile__("cpuid;"
                       : "=a"(RaxOut), "=b"(RbxOut), "=c"(RcxOut), "=d"(RdxOut)
                       : "a"(*Rax), "b"(*Rbx), "c"(*Rcx), "d"(*Rdx));
  *Rax = RaxOut;
  *Rbx = RbxOut;
  *Rcx = RcxOut;
  *Rdx = RdxOut;
#else
  std::array<int, 4> cpuid;

  __cpuidex(cpuid.data(), *Rax, *Rcx);

  *Rax = cpuid[0];
  *Rbx = cpuid[1];
  *Rcx = cpuid[2];
  *Rdx = cpuid[3];
#endif
}

} // namespace firestarter::environment::x86