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

#pragma once

#include <firestarter/Constants.hpp>
#include <firestarter/Environment/Environment.hpp>

#include <mutex>

namespace firestarter {

class LoadWorkerData {
public:
  LoadWorkerData(int id, environment::Environment &environment,
                 volatile unsigned long long *loadVar,
                 unsigned long long period, bool dumpRegisters)
      : addrHigh(loadVar), period(period), dumpRegisters(dumpRegisters),
        _id(id), _environment(environment),
        _config(new environment::platform::RuntimeConfig(
            environment.selectedConfig())) {}

  ~LoadWorkerData() { delete _config; }

  int id() const { return _id; }
  environment::Environment &environment() const { return _environment; }
  environment::platform::RuntimeConfig &config() const { return *_config; }

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
  bool dumpRegisters;

private:
  int _id;
  environment::Environment &_environment;
  environment::platform::RuntimeConfig *_config;
};

} // namespace firestarter
