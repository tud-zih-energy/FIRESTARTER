#include <firestarter/Environment/X86/X86Environment.hpp>

#include <ctime>

using namespace firestarter::environment::x86;

#ifndef __APPLE__
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

  auto scalingGovernor = this->getScalingGovernor();
  if (nullptr == scalingGovernor) {
    return EXIT_FAILURE;
  }

  std::string governor = scalingGovernor->getBuffer().str();

  /* non invariant TSCs can be used if CPUs run at fixed frequency */
  if (!this->hasInvariantRdtsc() && governor.compare("performance") &&
      governor.compare("powersave")) {
    return Environment::getCpuClockrate();
  }

  min_measurements = 5;

  if (!this->hasRdtsc()) {
    return Environment::getCpuClockrate();
  }

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
      else if (clock < clockrate)
        clockrate = clock;
    }
    i += 2;
  } while (((time_diff) < 10000) || (num_measurements < min_measurements));

  this->_clockrate = clockrate;

  return EXIT_SUCCESS;
}
#else
// TODO: get clockrate from intel processor name
int X86Environment::getCpuClockrate(void) { return EXIT_SUCCESS; }
#endif

#ifdef __APPLE__
// use sysctl to detect the name
std::string X86Environment::getProcessorName(void) { return std::string(""); }
#else
std::string X86Environment::getProcessorName(void) {
  return Environment::getProcessorName();
}
#endif

std::string X86Environment::getVendor(void) {
  return std::string(this->cpuInfo.vendor());
}
