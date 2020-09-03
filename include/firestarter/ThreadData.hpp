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

#ifndef INCLUDE_FIRESTARTER_THREADDATA_HPP
#define INCLUDE_FIRESTARTER_THREADDATA_HPP

#define THREAD_WAIT 1
#define THREAD_WORK 2
#define THREAD_INIT 3
#define THREAD_STOP 4
#define THREAD_INIT_FAILURE 0xffffffff

/* DO NOT CHANGE! the asm load-loop tests if load-variable is == 0 */
#define LOAD_LOW 0
/* DO NOT CHANGE! the asm load-loop continues until the load-variable is != 1 */
#define LOAD_HIGH 1
#define LOAD_STOP 2

#include <firestarter/Environment/Environment.hpp>

#include <mutex>

namespace firestarter {

class ThreadData {
public:
  ThreadData(int id, environment::Environment *environment,
             volatile unsigned long long *loadVar, unsigned long long period)
      : addrHigh(loadVar), period(period), _id(id), _environment(environment),
        _config(
            new environment::platform::Config(*environment->selectedConfig)){};
  ~ThreadData(){};

  const int &id = _id;
  environment::Environment *const &environment = _environment;
  environment::platform::Config *const &config = _config;

  int comm = THREAD_WAIT;
  bool ack = false;
  std::mutex mutex;
  unsigned long long *addrMem;
  volatile unsigned long long *addrHigh;
  unsigned long long buffersizeMem;
  unsigned long long iterations = 0;
  unsigned long long flops;
  unsigned long long start_tsc;
  unsigned long long stop_tsc;
  // period in usecs
  // used in low load routine to sleep 1/100th of this time
  unsigned long long period;

private:
  int _id;
  environment::Environment *_environment;
  environment::platform::Config *_config;
};

} // namespace firestarter

#endif
