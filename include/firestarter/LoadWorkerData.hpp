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
#include <firestarter/DumpRegisterStruct.hpp>
#include <firestarter/Environment/Environment.hpp>
#include <firestarter/ErrorDetectionStruct.hpp>

#include <atomic>
#include <memory>
#include <mutex>

#define PAD_SIZE(size, align)                                                  \
  align *(int)std::ceil((double)size / (double)align)

#if defined(__APPLE__)
#define ALIGNED_MALLOC(size, align) aligned_alloc(align, PAD_SIZE(size, align))
#define ALIGNED_FREE free
#elif defined(__MINGW64__)
#define ALIGNED_MALLOC(size, align) _mm_malloc(PAD_SIZE(size, align), align)
#define ALIGNED_FREE _mm_free
#elif defined(_MSC_VER)
#define ALIGNED_MALLOC(size, align)                                            \
  _aligned_malloc(PAD_SIZE(size, align), align)
#define ALIGNED_FREE _aligned_free
#else
#define ALIGNED_MALLOC(size, align)                                            \
  std::aligned_alloc(align, PAD_SIZE(size, align))
#define ALIGNED_FREE std::free
#endif

namespace firestarter {

class LoadWorkerData {
public:
  LoadWorkerData(int id, environment::Environment &environment,
                 volatile unsigned long long *loadVar,
                 unsigned long long period, bool dumpRegisters,
                 bool errorDetection)
      : addrHigh(loadVar), period(period), dumpRegisters(dumpRegisters),
        errorDetection(errorDetection), _id(id), _environment(environment),
        _config(new environment::platform::RuntimeConfig(
            environment.selectedConfig())) {
    // use REGISTER_MAX_NUM cache lines for the dumped registers
    // and another cache line for the control variable.
    // as we are doing aligned moves we only have the option to waste a whole
    // cacheline
    addrOffset = dumpRegisters
                     ? sizeof(DumpRegisterStruct) / sizeof(unsigned long long)
                     : 0;

    addrOffset += errorDetection ? sizeof(ErrorDetectionStruct) /
                                       sizeof(unsigned long long)
                                 : 0;
  }

  ~LoadWorkerData() {
    delete _config;
    if (addrMem - addrOffset != nullptr) {
      ALIGNED_FREE(addrMem - addrOffset);
    }
  }

  void setErrorCommunication(
      std::shared_ptr<unsigned long long> communicationLeft,
      std::shared_ptr<unsigned long long> communicationRight) {
    this->communicationLeft = communicationLeft;
    this->communicationRight = communicationRight;
  }

  int id() const { return _id; }
  environment::Environment &environment() const { return _environment; }
  environment::platform::RuntimeConfig &config() const { return *_config; }

  const ErrorDetectionStruct *errorDetectionStruct() const {
    return reinterpret_cast<ErrorDetectionStruct *>(addrMem - addrOffset);
  }

  int comm = THREAD_WAIT;
  bool ack = false;
  std::mutex mutex;
  unsigned long long *addrMem = nullptr;
  unsigned long long addrOffset;
  volatile unsigned long long *addrHigh;
  unsigned long long buffersizeMem;
  unsigned long long iterations = 0;
  // save the last iteration count when switching payloads
  std::atomic<unsigned long long> lastIterations;
  unsigned long long flops;
  unsigned long long startTsc;
  unsigned long long stopTsc;
  std::atomic<unsigned long long> lastStartTsc;
  std::atomic<unsigned long long> lastStopTsc;
  // period in usecs
  // used in low load routine to sleep 1/100th of this time
  unsigned long long period;
  bool dumpRegisters;
  bool errorDetection;
  std::shared_ptr<unsigned long long> communicationLeft;
  std::shared_ptr<unsigned long long> communicationRight;

private:
  int _id;
  environment::Environment &_environment;
  environment::platform::RuntimeConfig *_config;
};

} // namespace firestarter
