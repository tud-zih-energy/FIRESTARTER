/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2021 TU Dresden, Center for Information Services and High
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

using namespace firestarter;

int Firestarter::watchdogWorker(std::chrono::microseconds period,
                                std::chrono::microseconds load,
                                std::chrono::seconds timeout) {

  using clock = std::chrono::high_resolution_clock;
  using nsec = std::chrono::nanoseconds;
  using usec = std::chrono::microseconds;
  using sec = std::chrono::seconds;

  // calculate idle time to be the rest of the period
  auto idle = period - load;

  // elapsed time
  nsec time(0);

  // do no enter the loop if we do not have to set the load level periodically,
  // at 0 or 100 load.
  if (period > usec::zero()) {
    // this first time is critical as the period will be alligend from this
    // point
    std::chrono::time_point<clock> startTime = clock::now();

    // this loop will set the load level periodically.
    for (;;) {
      std::chrono::time_point<clock> currentTime = clock::now();

      // get the time already advanced in the current timeslice
      // this can happen if a load function does not terminates just on time
      nsec advance = std::chrono::duration_cast<nsec>(currentTime - startTime) %
                     std::chrono::duration_cast<nsec>(period);

      // subtract the advaned time from our timeslice by spilting it based on
      // the load level
      nsec load_reduction =
          (std::chrono::duration_cast<nsec>(load).count() * advance) /
          std::chrono::duration_cast<nsec>(period).count();
      nsec idle_reduction = advance - load_reduction;

      // signal high load level
      this->setLoad(LOAD_HIGH);

      // calculate values for nanosleep
      nsec load_nsec = load - load_reduction;

      // wait for time to be ellapsed with high load
#ifdef ENABLE_VTRACING
      VT_USER_START("WD_HIGH");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_BEGIN("WD_HIGH",
                                       SCOREP_USER_REGION_TYPE_COMMON);
#endif
      {
        std::unique_lock<std::mutex> lk(this->_watchdogTerminateMutex);
        // abort waiting if we get the interrupt signal
        this->_watchdogTerminateAlert.wait_for(
            lk, load_nsec, [this]() { return this->_watchdog_terminate; });
        // terminate on interrupt
        if (this->_watchdog_terminate) {
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
      this->setLoad(LOAD_LOW);

      // calculate values for nanosleep
      nsec idle_nsec = idle - idle_reduction;

      // wait for time to be ellapsed with low load
#ifdef ENABLE_VTRACING
      VT_USER_START("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_BEGIN("WD_LOW",
                                       SCOREP_USER_REGION_TYPE_COMMON);
#endif
      {
        std::unique_lock<std::mutex> lk(this->_watchdogTerminateMutex);
        // abort waiting if we get the interrupt signal
        this->_watchdogTerminateAlert.wait_for(
            lk, idle_nsec, [this]() { return this->_watchdog_terminate; });
        // terminate on interrupt
        if (this->_watchdog_terminate) {
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
      time += period;

      // exit when termination signal is received or timeout is reached
      {
        std::lock_guard<std::mutex> lk(this->_watchdogTerminateMutex);
        if (this->_watchdog_terminate ||
            (timeout > sec::zero() && (time > timeout))) {
          this->setLoad(LOAD_STOP);

          return EXIT_SUCCESS;
        }
      }
    }
  }

  // if timeout is set, sleep for this time and stop execution.
  // else return and wait for sigterm handler to request threads to stop.
  if (timeout > sec::zero()) {
    {
      std::unique_lock<std::mutex> lk(Firestarter::_watchdogTerminateMutex);
      // abort waiting if we get the interrupt signal
      Firestarter::_watchdogTerminateAlert.wait_for(
          lk, timeout, []() { return Firestarter::_watchdog_terminate; });
    }

    this->setLoad(LOAD_STOP);

    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}
