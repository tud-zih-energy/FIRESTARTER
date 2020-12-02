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

#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>

#ifdef ENABLE_VTRACING
#include <vt_user.h>
#endif
#ifdef ENABLE_SCOREP
#include <SCOREP_User.h>
#endif

#include <cmath>
#include <cstdlib>
#include <functional>
#include <thread>

#define PAD_SIZE(size, align)                                                  \
  align *(int)std::ceil((double)size / (double)align)

#if defined(__APPLE__)
#define ALIGNED_MALLOC(size, align) aligned_alloc(align, PAD_SIZE(size, align))
#define ALIGNED_FREE free
#elif defined(__MINGW64__)
#define ALIGNED_MALLOC(size, align) _mm_malloc(PAD_SIZE(size, align), align)
#define ALIGNED_FREE _mm_free
#else
#define ALIGNED_MALLOC(size, align)                                            \
  std::aligned_alloc(align, PAD_SIZE(size, align))
#define ALIGNED_FREE std::free
#endif

using namespace firestarter;

int Firestarter::initLoadWorkers(bool lowLoad, unsigned long long period,
                                 bool dumpRegisters) {
  int returnCode;

  if (EXIT_SUCCESS != (returnCode = this->environment->setCpuAffinity(0))) {
    return EXIT_FAILURE;
  }

  // setup load variable to execute low or high load once the threads switch to
  // work.
  this->loadVar = lowLoad ? LOAD_LOW : LOAD_HIGH;

  // allocate buffer for threads
  pthread_t *threads = nullptr;
  if (nullptr ==
      (threads = static_cast<pthread_t *>(ALIGNED_MALLOC(
           this->environment->requestedNumThreads * sizeof(pthread_t), 64)))) {
    log::error() << "Could not allocate pthread_t";
    return EXIT_FAILURE;
  }

  for (unsigned long long i = 0; i < this->environment->requestedNumThreads;
       i++) {
    auto td = new LoadWorkerData(i, this->environment, &this->loadVar, period,
                                 dumpRegisters);

    auto dataCacheSizeIt =
        td->config->platformConfig->dataCacheBufferSize.begin();
    auto ramBufferSize = td->config->platformConfig->ramBufferSize;

    td->buffersizeMem = (*dataCacheSizeIt + *std::next(dataCacheSizeIt, 1) +
                         *std::next(dataCacheSizeIt, 2) + ramBufferSize) /
                        td->config->thread / sizeof(unsigned long long);

    // create the thread
    if (EXIT_SUCCESS !=
        (returnCode = pthread_create(&threads[i], NULL, loadThreadWorker,
                                     std::ref(td)))) {
      log::error() << "pthread_create failed with returnCode " << returnCode;
      return EXIT_FAILURE;
    }

    this->loadThreads.push_back(std::make_pair(&threads[i], std::ref(td)));
  }

  this->signalLoadWorkers(THREAD_INIT);

  return EXIT_SUCCESS;
}

void Firestarter::signalLoadWorkers(int comm) {
  bool ack;

  // start the work
  for (auto const &thread : this->loadThreads) {
    auto td = thread.second;

    td->mutex.lock();
  }

  for (auto const &thread : this->loadThreads) {
    auto td = thread.second;

    td->comm = comm;
    td->mutex.unlock();
  }

  for (auto const &thread : this->loadThreads) {
    auto td = thread.second;

    do {
      td->mutex.lock();
      ack = td->ack;
      td->mutex.unlock();
    } while (!ack);

    td->mutex.lock();
    td->ack = false;
    td->mutex.unlock();
  }
}

void Firestarter::joinLoadWorkers() {
  // wait for threads after watchdog has requested termination
  for (auto const &thread : this->loadThreads) {
    pthread_join(*thread.first, NULL);
  }
}

void Firestarter::printPerformanceReport() {
  // performance report
  unsigned long long startTimestamp = 0xffffffffffffffff;
  unsigned long long stopTimestamp = 0;

  unsigned long long iterations = 0;

  log::debug() << "\nperformance report:\n";

  for (auto const &thread : this->loadThreads) {
    auto td = thread.second;

    log::debug() << "Thread " << td->id << ": " << td->iterations
                 << " iterations, tsc_delta: " << td->stop_tsc - td->start_tsc;

    if (startTimestamp > td->start_tsc) {
      startTimestamp = td->start_tsc;
    }
    if (stopTimestamp < td->stop_tsc) {
      stopTimestamp = td->stop_tsc;
    }

    iterations += td->iterations;
  }

  double runtime = (double)(stopTimestamp - startTimestamp) /
                   (double)this->environment->topology().clockrate();
  double gFlops =
      (double)this->loadThreads.front().second->config->payload->flops *
      0.000000001 * (double)iterations / runtime;
  double bandwidth =
      (double)this->loadThreads.front().second->config->payload->bytes *
      0.000000001 * (double)iterations / runtime;

  // format runtime, gflops and bandwidth %.2f
  const char *fmt = "%.2f";
  int size;

#define FORMAT(input)                                                          \
  size = std::snprintf(nullptr, 0, fmt, input);                                \
  std::vector<char> input##Vector(size + 1);                                   \
  std::snprintf(&input##Vector[0], input##Vector.size(), fmt, input);          \
  auto input##String = std::string(&input##Vector[0])

  FORMAT(runtime);
  FORMAT(gFlops);
  FORMAT(bandwidth);

#undef FORMAT

  log::debug()
      << "\n"
      << "total iterations: " << iterations << "\n"
      << "runtime: " << runtimeString << " seconds ("
      << stopTimestamp - startTimestamp << " cycles)\n"
      << "\n"
      << "estimated floating point performance: " << gFlopsString << " GFLOPS\n"
      << "estimated memory bandwidth*: " << bandwidthString << " GB/s\n"
      << "\n"
      << "* this estimate is highly unreliable if --function is used in order "
         "to "
         "select\n"
      << "  a function that is not optimized for your architecture, or if "
         "FIRESTARTER is\n"
      << "  executed on an unsupported architecture!";
}

void *Firestarter::loadThreadWorker(void *loadWorkerData) {

  int old = THREAD_WAIT;

  auto td = reinterpret_cast<LoadWorkerData *>(loadWorkerData);

  // use REGISTER_MAX_NUM cache lines for the dumped registers
  // and another cache line for the control variable.
  // as we are doing aligned moves we only have the option to waste a whole
  // cacheline
  unsigned long long addrOffset =
      td->dumpRegisters
          ? sizeof(DumpRegisterStruct) / sizeof(unsigned long long)
          : 0;

#ifndef __APPLE__
  pthread_setname_np(pthread_self(), "LoadWorker");
#endif

  for (;;) {
    td->mutex.lock();
    int comm = td->comm;
    td->mutex.unlock();

    if (comm != old) {
      old = comm;

      td->mutex.lock();
      td->ack = true;
      td->mutex.unlock();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      continue;
    }

    switch (comm) {
    // allocate and initialize memory
    case THREAD_INIT:
      // set affinity
      td->environment->setCpuAffinity(td->id);

      // compile payload
      td->config->payload->compilePayload(
          td->config->payloadSettings, td->config->instructionCacheSize,
          td->config->dataCacheBufferSize, td->config->ramBufferSize,
          td->config->thread, td->config->lines, td->dumpRegisters);

      // allocate memory
      // if we should dump some registers, we use the first part of the memory
      // for them.
      td->addrMem =
          reinterpret_cast<unsigned long long *>(ALIGNED_MALLOC(
              (td->buffersizeMem + addrOffset) * sizeof(unsigned long long),
              64)) +
          addrOffset;

      // TODO: handle error

      if (td->dumpRegisters) {
        reinterpret_cast<DumpRegisterStruct *>(td->addrMem - addrOffset)
            ->dumpVar = DumpVariable::Wait;
      }

      // call init function
      td->config->payload->init(td->addrMem, td->buffersizeMem);
      break;
    // perform stress test
    case THREAD_WORK:
      // record threads start timestamp
      td->start_tsc = td->environment->topology().timestamp();

      // will be terminated by watchdog
      for (;;) {
        // call high load function
#ifdef ENABLE_VTRACING
        VT_USER_START("HIGH_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_BEGIN("HIGH",
                                         SCOREP_USER_REGION_TYPE_COMMON);
#endif
        td->iterations = td->config->payload->highLoadFunction(
            td->addrMem, td->addrHigh, td->iterations);

        // call low load function
#ifdef ENABLE_VTRACING
        VT_USER_END("HIGH_LOAD_FUNC");
        VT_USER_START("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("HIGH");
        SCOREP_USER_REGION_BY_NAME_BEGIN("LOW", SCOREP_USER_REGION_TYPE_COMMON);
#endif
        td->config->payload->lowLoadFunction(td->addrHigh, td->period);
#ifdef ENABLE_VTRACING
        VT_USER_END("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("LOW");
#endif

        // terminate if master signals end of run and record stop timestamp
        if (*td->addrHigh == LOAD_STOP) {
          td->stop_tsc = td->environment->topology().timestamp();

          ALIGNED_FREE(td->addrMem - addrOffset);
          pthread_exit(NULL);
        }
      }
      break;
    case THREAD_WAIT:
      break;
    case THREAD_STOP:
    default:
      ALIGNED_FREE(td->addrMem - addrOffset);
      pthread_exit(0);
    }
  }
}
