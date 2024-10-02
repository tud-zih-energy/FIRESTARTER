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

#include <firestarter/Firestarter.hpp>

#include <cerrno>
#include <csignal>

#ifdef ENABLE_SCOREP
#include <SCOREP_User.h>
#endif

namespace firestarter {

auto Firestarter::watchdogWorker(std::chrono::microseconds Period, std::chrono::microseconds Load,
                                 std::chrono::seconds Timeout) -> int {

  using clock = std::chrono::high_resolution_clock;
  using nsec = std::chrono::nanoseconds;
  using usec = std::chrono::microseconds;
  using sec = std::chrono::seconds;

  // calculate idle time to be the rest of the period
  auto Idle = Period - Load;

  // elapsed time
  nsec Time(0);

  // do no enter the loop if we do not have to set the load level periodically,
  // at 0 or 100 load.
  if (Period > usec::zero()) {
    // this first time is critical as the period will be alligend from this
    // point
    std::chrono::time_point<clock> StartTime = clock::now();

    // this loop will set the load level periodically.
    for (;;) {
      std::chrono::time_point<clock> CurrentTime = clock::now();

      // get the time already advanced in the current timeslice
      // this can happen if a load function does not terminates just on time
      nsec Advance =
          std::chrono::duration_cast<nsec>(CurrentTime - StartTime) % std::chrono::duration_cast<nsec>(Period);

      // subtract the advaned time from our timeslice by spilting it based on
      // the load level
      nsec LoadReduction =
          (std::chrono::duration_cast<nsec>(Load).count() * Advance) / std::chrono::duration_cast<nsec>(Period).count();
      nsec IdleReduction = Advance - LoadReduction;

      // signal high load level
      setLoad(LOAD_HIGH);

      // calculate values for nanosleep
      nsec LoadNsec = Load - LoadReduction;

      // wait for time to be ellapsed with high load
#ifdef ENABLE_VTRACING
      VT_USER_START("WD_HIGH");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_BEGIN("WD_HIGH", SCOREP_USER_REGION_TYPE_COMMON);
#endif
      {
        std::unique_lock<std::mutex> Lk(WatchdogTerminateMutex);
        // abort waiting if we get the interrupt signal
        WatchdogTerminateAlert.wait_for(Lk, LoadNsec, []() { return WatchdogTerminate; });
        // terminate on interrupt
        if (WatchdogTerminate) {
          return EXIT_SUCCESS;
        }
      }
#ifdef ENABLE_VTRACING
      VT_USER_END("WD_HIGH");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_END("WD_HIGH");
#endif

      // signal low load
      setLoad(LOAD_LOW);

      // calculate values for nanosleep
      nsec IdleNsec = Idle - IdleReduction;

      // wait for time to be ellapsed with low load
#ifdef ENABLE_VTRACING
      VT_USER_START("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_BEGIN("WD_LOW", SCOREP_USER_REGION_TYPE_COMMON);
#endif
      {
        std::unique_lock<std::mutex> Lk(WatchdogTerminateMutex);
        // abort waiting if we get the interrupt signal
        WatchdogTerminateAlert.wait_for(Lk, IdleNsec, []() { return WatchdogTerminate; });
        // terminate on interrupt
        if (WatchdogTerminate) {
          return EXIT_SUCCESS;
        }
      }
#ifdef ENABLE_VTRACING
      VT_USER_END("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_END("WD_LOW");
#endif

      // increment elapsed time
      Time += Period;

      // exit when termination signal is received or timeout is reached
      {
        std::lock_guard<std::mutex> Lk(WatchdogTerminateMutex);
        if (WatchdogTerminate || (Timeout > sec::zero() && (Time > Timeout))) {
          setLoad(LOAD_STOP);

          return EXIT_SUCCESS;
        }
      }
    }
  }

  // if timeout is set, sleep for this time and stop execution.
  // else return and wait for sigterm handler to request threads to stop.
  if (Timeout > sec::zero()) {
    {
      std::unique_lock<std::mutex> Lk(Firestarter::WatchdogTerminateMutex);
      // abort waiting if we get the interrupt signal
      Firestarter::WatchdogTerminateAlert.wait_for(Lk, Timeout, []() { return WatchdogTerminate; });
    }

    setLoad(LOAD_STOP);

    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}

} // namespace firestarter