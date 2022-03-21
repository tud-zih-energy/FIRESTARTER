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

#include <firestarter/ErrorDetectionStruct.hpp>
#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>

#if defined(linux) || defined(__linux__)
extern "C" {
#include <firestarter/Measurement/Metric/IPCEstimate.h>
}
#endif

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

using namespace firestarter;

auto aligned_free_deleter = [](void *p) { ALIGNED_FREE(p); };

int Firestarter::initLoadWorkers(bool lowLoad, unsigned long long period) {
  int returnCode;

  if (EXIT_SUCCESS != (returnCode = this->environment().setCpuAffinity(0))) {
    return EXIT_FAILURE;
  }

  // setup load variable to execute low or high load once the threads switch to
  // work.
  this->loadVar = lowLoad ? LOAD_LOW : LOAD_HIGH;

  auto numThreads = this->environment().requestedNumThreads();

  // create a std::vector<std::shared_ptr<>> of requestenNumThreads()
  // communication pointers and add these to the threaddata
  if (_errorDetection) {
    for (unsigned long long i = 0; i < numThreads; i++) {
      auto commPtr = reinterpret_cast<unsigned long long *>(
          ALIGNED_MALLOC(2 * sizeof(unsigned long long), 64));
      assert(commPtr);
      this->errorCommunication.push_back(
          std::shared_ptr<unsigned long long>(commPtr, aligned_free_deleter));
      log::debug() << "Threads " << (i + numThreads - 1) % numThreads << " and "
                   << i << " commPtr = 0x" << std::setfill('0')
                   << std::setw(sizeof(unsigned long long) * 2) << std::hex
                   << (unsigned long long)commPtr;
    }
  }

  for (unsigned long long i = 0; i < numThreads; i++) {
    auto td = std::make_shared<LoadWorkerData>(i, this->environment(),
                                               &this->loadVar, period,
                                               _dumpRegisters, _errorDetection);

    if (_errorDetection) {
      // distribute pointers for error deteciton. (set threads in a ring)
      // give this thread the left pointer i and right pointer (i+1) %
      // requestedNumThreads().
      td->setErrorCommunication(this->errorCommunication[i],
                                this->errorCommunication[(i + 1) % numThreads]);
    }

    auto dataCacheSizeIt =
        td->config().platformConfig().dataCacheBufferSize().begin();
    auto ramBufferSize = td->config().platformConfig().ramBufferSize();

    td->buffersizeMem = (*dataCacheSizeIt + *std::next(dataCacheSizeIt, 1) +
                         *std::next(dataCacheSizeIt, 2) + ramBufferSize) /
                        td->config().thread() / sizeof(unsigned long long);

    // create the thread
    std::thread t(Firestarter::loadThreadWorker, td);

    log::trace() << "Created thread #" << i << " with ID: " << t.get_id();

    if (i == 0) {
      // only show error for all worker threads except first.
      firestarter::logging::FirstWorkerThreadFilter<
          firestarter::logging::record>::setFirstThread(t.get_id());
    }

    this->loadThreads.push_back(std::make_pair(std::move(t), td));
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
  for (auto &thread : this->loadThreads) {
    thread.first.join();
  }
}

void Firestarter::printThreadErrorReport() {
  if (_errorDetection) {
    auto maxSize = this->loadThreads.size();

    std::vector<bool> errors(maxSize, false);

    for (decltype(maxSize) i = 0; i < maxSize; i++) {
      auto errorDetectionStruct =
          this->loadThreads[i].second->errorDetectionStruct();

      if (errorDetectionStruct->errorLeft) {
        errors[(i + maxSize - 1) % maxSize] = true;
      }
      if (errorDetectionStruct->errorRight) {
        errors[i] = true;
      }
    }

    for (decltype(maxSize) i = 0; i < maxSize; i++) {
      if (errors[i]) {
        log::fatal()
            << "Data mismatch between Threads " << i << " and "
            << (i + 1) % maxSize
            << ".\n       This may be caused by bit-flips in the hardware.";
      }
    }
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

    log::debug() << "Thread " << td->id() << ": " << td->iterations
                 << " iterations, tsc_delta: " << td->stopTsc - td->startTsc;

    if (startTimestamp > td->startTsc) {
      startTimestamp = td->startTsc;
    }
    if (stopTimestamp < td->stopTsc) {
      stopTimestamp = td->stopTsc;
    }

    iterations += td->iterations;
  }

  double runtime = (double)(stopTimestamp - startTimestamp) /
                   (double)this->environment().topology().clockrate();
  double gFlops =
      (double)this->loadThreads.front().second->config().payload().flops() *
      0.000000001 * (double)iterations / runtime;
  double bandwidth =
      (double)this->loadThreads.front().second->config().payload().bytes() *
      0.000000001 * (double)iterations / runtime;

  // insert values for ipc-estimate metric
  // if we are on linux
#if defined(linux) || defined(__linux__)
  if (_measurement) {
    for (auto const &thread : this->loadThreads) {
      auto td = thread.second;
      ipc_estimate_metric_insert((double)td->iterations *
                                 (double)this->loadThreads.front()
                                     .second->config()
                                     .payload()
                                     .instructions() /
                                 (double)(stopTimestamp - startTimestamp));
    }
  }
#endif

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

void Firestarter::loadThreadWorker(std::shared_ptr<LoadWorkerData> td) {

  int old = THREAD_WAIT;

#if defined(linux) || defined(__linux__)
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
      td->environment().setCpuAffinity(td->id());

      // compile payload
      td->config().payload().compilePayload(
          td->config().payloadSettings(), td->config().instructionCacheSize(),
          td->config().dataCacheBufferSize(), td->config().ramBufferSize(),
          td->config().thread(), td->config().lines(), td->dumpRegisters,
          td->errorDetection);

      // allocate memory
      // if we should dump some registers, we use the first part of the memory
      // for them.
      td->addrMem =
          reinterpret_cast<unsigned long long *>(ALIGNED_MALLOC(
              (td->buffersizeMem + td->addrOffset) * sizeof(unsigned long long),
              64)) +
          td->addrOffset;

      // exit application on error
      if (td->addrMem - td->addrOffset == nullptr) {
        workerLog::error() << "Could not allocate memory for CPU load thread "
                           << td->id() << "\n";
        exit(ENOMEM);
      }

      if (td->dumpRegisters) {
        reinterpret_cast<DumpRegisterStruct *>(td->addrMem - td->addrOffset)
            ->dumpVar = DumpVariable::Wait;
      }

      if (td->errorDetection) {
        auto errorDetectionStruct = reinterpret_cast<ErrorDetectionStruct *>(
            td->addrMem - td->addrOffset);

        std::memset(errorDetectionStruct, 0, sizeof(ErrorDetectionStruct));

        // distribute left and right communication pointers
        errorDetectionStruct->communicationLeft = td->communicationLeft.get();
        errorDetectionStruct->communicationRight = td->communicationRight.get();

        // do first touch memset 0 for the communication pointers
        std::memset((void *)errorDetectionStruct->communicationLeft, 0,
                    sizeof(unsigned long long) * 2);
      }

      // call init function
      td->config().payload().init(td->addrMem, td->buffersizeMem);

      break;
    // perform stress test
    case THREAD_WORK:
      // record threads start timestamp
      td->startTsc = td->environment().topology().timestamp();

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
        td->iterations = td->config().payload().highLoadFunction(
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
        td->config().payload().lowLoadFunction(td->addrHigh, td->period);
#ifdef ENABLE_VTRACING
        VT_USER_END("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("LOW");
#endif

        // terminate if master signals end of run and record stop timestamp
        if (*td->addrHigh == LOAD_STOP) {
          td->stopTsc = td->environment().topology().timestamp();

          return;
        }

        if (*td->addrHigh == LOAD_SWITCH) {
          td->stopTsc = td->environment().topology().timestamp();

          break;
        }
      }
      break;
    case THREAD_SWITCH:
      // compile payload
      td->config().payload().compilePayload(
          td->config().payloadSettings(), td->config().instructionCacheSize(),
          td->config().dataCacheBufferSize(), td->config().ramBufferSize(),
          td->config().thread(), td->config().lines(), td->dumpRegisters,
          td->errorDetection);

      // call init function
      td->config().payload().init(td->addrMem, td->buffersizeMem);

      // save old iteration count
      td->lastIterations = td->iterations;
      td->lastStartTsc = td->startTsc;
      td->lastStopTsc = td->stopTsc;
      td->iterations = 0;
      break;
    case THREAD_WAIT:
      break;
    case THREAD_STOP:
    default:
      return;
    }
  }
}
