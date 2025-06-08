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

#include "firestarter/Constants.hpp"
#include "firestarter/Firestarter.hpp"
#include "firestarter/Tracing.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <mutex>

namespace firestarter {

void Firestarter::watchdogWorker(std::chrono::microseconds Period, std::chrono::microseconds Load,
                                 std::chrono::seconds Timeout) {

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
    const auto StartTime = clock::now();

    // this loop will set the load level periodically.
    for (;;) {
      const auto CurrentTime = clock::now();

      // get the time already advanced in the current timeslice
      // this can happen if a load function does not terminates just on time
      const auto Advance =
          std::chrono::duration_cast<nsec>(CurrentTime - StartTime) % std::chrono::duration_cast<nsec>(Period);

      // subtract the advaned time from our timeslice by spilting it based on
      // the load level
      const auto LoadReduction =
          (std::chrono::duration_cast<nsec>(Load).count() * Advance) / std::chrono::duration_cast<nsec>(Period).count();
      const auto IdleReduction = Advance - LoadReduction;

      // signal high load level
      setLoad(LoadThreadWorkType::LoadHigh);

      // calculate values for nanosleep
      const auto LoadNsec = Load - LoadReduction;

      // wait for time to be ellapsed with high load
      firestarterTracingRegionBegin("WD_HIGH");
      {
        std::unique_lock<std::mutex> Lk(WatchdogTerminateMutex);
        // abort waiting if we get the interrupt signal
        WatchdogTerminateAlert.wait_for(Lk, LoadNsec, []() { return WatchdogTerminate; });
        // terminate on interrupt
        if (WatchdogTerminate) {
          return;
        }
      }
      firestarterTracingRegionEnd("WD_HIGH");

      // signal low load
      setLoad(LoadThreadWorkType::LoadLow);

      // calculate values for nanosleep
      const auto IdleNsec = Idle - IdleReduction;

      // wait for time to be ellapsed with low load
      firestarterTracingRegionBegin("WD_LOW");
      {
        std::unique_lock<std::mutex> Lk(WatchdogTerminateMutex);
        // abort waiting if we get the interrupt signal
        WatchdogTerminateAlert.wait_for(Lk, IdleNsec, []() { return WatchdogTerminate; });
        // terminate on interrupt
        if (WatchdogTerminate) {
          return;
        }
      }
      firestarterTracingRegionEnd("WD_LOW");

      // increment elapsed time
      Time += Period;

      // exit when termination signal is received or timeout is reached
      {
        const std::lock_guard<std::mutex> Lk(WatchdogTerminateMutex);
        if (WatchdogTerminate || (Timeout > sec::zero() && (Time > Timeout))) {
          setLoad(LoadThreadWorkType::LoadStop);

          return;
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

    setLoad(LoadThreadWorkType::LoadStop);
  }
}

} // namespace firestarter