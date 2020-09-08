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

#include <firestarter/Environment/X86/X86Environment.hpp>

#include <ctime>
#include <array>

using namespace firestarter::environment::x86;

// measures clockrate using the Time-Stamp-Counter
// only constant TSCs will be used (i.e. power management indepent TSCs)
// save frequency in highest P-State or use generic fallback if no invarient TSC
// is available
int X86Environment::getCpuClockrate(void) {
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::microseconds ticks;

  unsigned long long start1_tsc, start2_tsc, end1_tsc, end2_tsc;
  unsigned long long time_diff;
  unsigned long long clock_lower_bound, clock_upper_bound, clock;
  unsigned long long clockrate = 0;
  int i, num_measurements = 0, min_measurements;

  Clock::time_point start_time, end_time;

#if not(defined(__APPLE__) || defined(_WIN32))
  auto governor = this->getScalingGovernor().str();
  if (governor.empty()) {
    return EXIT_FAILURE;
  }

  /* non invariant TSCs can be used if CPUs run at fixed frequency */
  if (!this->hasInvariantRdtsc() && governor.compare("performance") &&
      governor.compare("powersave")) {
    return Environment::getCpuClockrate();
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

  this->_clockrate = clockrate;

  return EXIT_SUCCESS;
}

#ifdef __APPLE__
// use sysctl to detect the name
std::string X86Environment::getProcessorName(void) {
  std::array<char, 128> buffer;
  auto cmd = "sysctl -n machdep.cpu.brand_string";
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    log::warn() << "Could not determine processor-name";
  }
  if (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    auto str = std::string(buffer.data());
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    return str;
  } else {
    return "";
  }
}
#elif defined(_WIN32)
// use wmic
std::string X86Environment::getProcessorName(void) {
  std::array<char, 128> buffer;
  auto cmd = "wmic cpu get name";
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    log::warn() << "Could not determine processor-name";
  }
  auto line = 0;
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    if (line != 1) {
      line++;
      continue;
    }

    auto str = std::string(buffer.data());
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    return str;
  }
  return "";
}
#else
std::string X86Environment::getProcessorName(void) {
  return Environment::getProcessorName();
}
#endif

std::string X86Environment::getVendor(void) {
  return std::string(this->cpuInfo.vendor());
}

std::list<std::string> X86Environment::getCpuFeatures(void) {
  std::list<std::string> featureList;

  for (int i = 0; i < (int)asmjit::x86::Features::kMaxFeatures; i++) {
    if (!this->cpuFeatures.has(i)) {
      continue;
    }

    asmjit::String sb;

    auto error = asmjit::Formatter::formatFeature(sb, this->cpuInfo.arch(), i);
    if (error != asmjit::ErrorCode::kErrorOk) {
      log::warn() << "Formatting cpu features got asmjit error: " << error;
    }

    featureList.push_back(std::string(sb.data()));
  }

  return featureList;
}
