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

#ifndef INCLUDE_FIRESTARTER_FIRESTARTER_HPP
#define INCLUDE_FIRESTARTER_FIRESTARTER_HPP

#ifdef BUILD_CUDA
#include <firestarter/Cuda/Cuda.hpp>
#endif

#include <firestarter/DumpRegisterWorkerData.hpp>
#include <firestarter/LoadWorkerData.hpp>

#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) ||            \
    defined(_M_X64)
#include <firestarter/Environment/X86/X86Environment.hpp>
#endif

#include <chrono>
#include <list>
#include <string>
#include <utility>

extern "C" {
#include <pthread.h>
}

namespace firestarter {

class Firestarter {
public:
  Firestarter(void) {
#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) ||            \
    defined(_M_X64)
    _environment = new environment::x86::X86Environment();
#else
#error "FIRESTARTER is not implemented for this ISA"
#endif

#ifdef BUILD_CUDA
    this->_gpuStructPointer = reinterpret_cast<cuda::gpustruct_t *>(
        malloc(sizeof(cuda::gpustruct_t)));
    this->_gpuStructPointer->loadingdone = 0;
    this->_gpuStructPointer->loadvar = &this->loadVar;
#endif
  };

  ~Firestarter(void) {
    delete _environment;

#ifdef BUILD_CUDA
    free(this->gpuStructPointer);
#endif
  };

  environment::Environment *const &environment = _environment;

#ifdef BUILD_CUDA
  cuda::gpustruct_t *const &gpuStructPointer = _gpuStructPointer;
#endif

  // LoadThreadWorker.cpp
  int initLoadWorkers(bool lowLoad, unsigned long long period,
                      bool dumpRegisters);
  void joinLoadWorkers(void);
  void printPerformanceReport(void);

  void signalWork(void) { signalLoadWorkers(THREAD_WORK); };

  // WatchdogWorker.cpp
  int watchdogWorker(std::chrono::microseconds period,
                     std::chrono::microseconds load,
                     std::chrono::seconds timeout);

  // DumpRegisterWorker.cpp
  int initDumpRegisterWorker(std::chrono::seconds dumpTimeDelta);
  void joinDumpRegisterWorker(void);

private:
  environment::Environment *_environment;

#ifdef BUILD_CUDA
  cuda::gpustruct_t *_gpuStructPointer;
#endif

  // LoadThreadWorker.cpp
  void signalLoadWorkers(int comm);
  static void *loadThreadWorker(void *loadWorkerData);

  // CudaWorker.cpp
  static void *cudaWorker(void *cudaData);

  // DumpRegisterWorker.cpp
  static void *dumpRegisterWorker(void *dumpRegisterWorkerData);

  // variable to control the load of the threads
  volatile unsigned long long loadVar = LOAD_LOW;

  std::list<std::pair<pthread_t *, LoadWorkerData *>> loadThreads;

  pthread_t dumpRegisterWorkerThread;
};

} // namespace firestarter

#endif
