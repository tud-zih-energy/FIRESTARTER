/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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

using namespace firestarter::environment::x86;

X86CPUTopology::X86CPUTopology()
    : CPUTopology("x86_64"), cpuInfo(asmjit::CpuInfo::host()),
      cpuFeatures(cpuInfo.features<asmjit::x86::Features>()),
      _vendor(this->cpuInfo.vendor()) {

  std::stringstream ss;
  ss << "Family " << this->familyId() << ", Model " << this->modelId()
     << ", Stepping " << this->stepping();
  this->_model = ss.str();

  for (int i = 0; i < (int)asmjit::x86::Features::kMaxFeatures; i++) {
    if (!this->cpuFeatures.has(i)) {
      continue;
    }

    asmjit::String sb;

    auto error = asmjit::Formatter::formatFeature(sb, this->cpuInfo.arch(), i);
    if (error != asmjit::ErrorCode::kErrorOk) {
      log::warn() << "Formatting cpu features got asmjit error: " << error;
    }

    this->featureList.push_back(std::string(sb.data()));
  }

  unsigned long long a = 0, b = 0, c = 0, d = 0;

  // check if we have rdtsc
  this->cpuid(&a, &b, &c, &d);
  if (a >= 1) {
    a = 1;
    this->cpuid(&a, &b, &c, &d);
    if ((int)d & (1 << 4)) {
      this->_hasRdtsc = true;
    } else {
      this->_hasRdtsc = false;
    }
  }

  // check if we have invarant rdtsc
  if (this->hasRdtsc()) {
    a = 0, b = 0, c = 0, d = 0;

    this->_hasInvariantRdtsc = true;

    /* TSCs are usable if CPU supports only one frequency in C0 (no
       speedstep/Cool'n'Quite)
       or if multiple frequencies are available and the constant/invariant TSC
       feature flag is set */

    if (0 == this->vendor().compare("INTEL")) {
      /*check if Powermanagement and invariant TSC are supported*/
      a = 1;
      this->cpuid(&a, &b, &c, &d);
      /* no Frequency control */
      if ((!(d & (1 << 22))) && (!(c & (1 << 7)))) {
        this->_hasInvariantRdtsc = true;
      } else {
        a = 0x80000000;
        this->cpuid(&a, &b, &c, &d);
        if (a >= 0x80000007) {
          a = 0x80000007;
          this->cpuid(&a, &b, &c, &d);
          /* invariant TSC */
          if (d & (1 << 8)) {
            this->_hasInvariantRdtsc = true;
          }
        }
      }
    }

    if (0 == this->vendor().compare("AMD")) {
      /*check if Powermanagement and invariant TSC are supported*/
      a = 0x80000000;
      this->cpuid(&a, &b, &c, &d);
      if (a >= 0x80000007) {
        a = 0x80000007;
        this->cpuid(&a, &b, &c, &d);

        /* no Frequency control */
        if ((!(d & (1 << 7))) && (!(d & (1 << 1)))) {
          this->_hasInvariantRdtsc = true;
        }
        /* invariant TSC */
        if (d & (1 << 8)) {
          this->_hasInvariantRdtsc = true;
        }
      }
      /* assuming no frequency control if cpuid does not provide the extended
         function to test for it */
      else {
        this->_hasInvariantRdtsc = true;
      }
    }
  }
}

// measures clockrate using the Time-Stamp-Counter
// only constant TSCs will be used (i.e. power management indepent TSCs)
// save frequency in highest P-State or use generic fallback if no invarient TSC
// is available
unsigned long long X86CPUTopology::clockrate() const {
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::microseconds ticks;

  unsigned long long start1_tsc, start2_tsc, end1_tsc, end2_tsc;
  unsigned long long time_diff;
  unsigned long long clock_lower_bound, clock_upper_bound, clock;
  unsigned long long clockrate = 0;
  int i, num_measurements = 0, min_measurements;

  Clock::time_point start_time, end_time;

#if not(defined(__APPLE__) || defined(_WIN32))
  auto governor = this->scalingGovernor();
  if (governor.empty()) {
    return CPUTopology::clockrate();
  }

  /* non invariant TSCs can be used if CPUs run at fixed frequency */
  if (!this->hasInvariantRdtsc() && governor.compare("performance") &&
      governor.compare("powersave")) {
    return CPUTopology::clockrate();
  }

  min_measurements = 5;
#else
  min_measurements = 20;
#endif

  i = 3;

  do {
    // start timestamp
    start1_tsc = this->timestamp();
    start_time = Clock::now();
    start2_tsc = this->timestamp();

    // waiting
    do {
      end1_tsc = this->timestamp();
    } while (end1_tsc < start2_tsc + 1000000 * i); /* busy waiting */

    // end timestamp
    do {
      end1_tsc = this->timestamp();
      end_time = Clock::now();
      end2_tsc = this->timestamp();

      time_diff =
          std::chrono::duration_cast<ticks>(end_time - start_time).count();
    } while (0 == time_diff);

    clock_lower_bound = (((end1_tsc - start2_tsc) * 1000000) / (time_diff));
    clock_upper_bound = (((end2_tsc - start1_tsc) * 1000000) / (time_diff));

    // if both values differ significantly, the measurement could have been
    // interrupted between 2 rdtsc's
    if (((double)clock_lower_bound > (((double)clock_upper_bound) * 0.999)) &&
        ((time_diff) > 2000)) {
      num_measurements++;
      clock = (clock_lower_bound + clock_upper_bound) / 2;
      if (clockrate == 0)
        clockrate = clock;
#ifndef _WIN32
      else if (clock < clockrate)
        clockrate = clock;
#else
      else if (clock > clockrate)
        clockrate = clock;
#endif
    }
    i += 2;
  } while (((time_diff) < 10000) || (num_measurements < min_measurements));

  return clockrate;
}

unsigned long long X86CPUTopology::timestamp() const {
#ifndef _MSC_VER
  unsigned long long reg_a, reg_d;
#else
  unsigned long long i;
#endif

  if (!this->hasRdtsc()) {
    return 0;
  }

#ifndef _MSC_VER
  __asm__ __volatile__("rdtsc;" : "=a"(reg_a), "=d"(reg_d));
  return (reg_d << 32) | (reg_a & 0xffffffffULL);
#else
  i = __rdtsc();
  return i;
#endif
}

void X86CPUTopology::cpuid(unsigned long long *a, unsigned long long *b,
                           unsigned long long *c, unsigned long long *d) const {
#ifndef _MSC_VER
  unsigned long long reg_a, reg_b, reg_c, reg_d;

  __asm__ __volatile__("cpuid;"
                       : "=a"(reg_a), "=b"(reg_b), "=c"(reg_c), "=d"(reg_d)
                       : "a"(*a), "b"(*b), "c"(*c), "d"(*d));
  *a = reg_a;
  *b = reg_b;
  *c = reg_c;
  *d = reg_d;
#else
  std::array<int, 4> cpuid;

  __cpuidex(cpuid.data(), *a, *c);

  *a = cpuid[0];
  *b = cpuid[1];
  *c = cpuid[2];
  *d = cpuid[3];
#endif
}
