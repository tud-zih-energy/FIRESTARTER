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

#include <firestarter/ErrorDetectionStruct.hpp>
#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>
#include <iomanip>

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

auto aligned_free_deleter = [](void* p) { ALIGNED_FREE(p); };

int Firestarter::initLoadWorkers(bool lowLoad, uint64_t period) {
  int returnCode;

  if (EXIT_SUCCESS != (returnCode = this->environment().setCpuAffinity(0))) {
    return EXIT_FAILURE;
  }

  // setup load variable to execute low or high load once the threads switch to
  // work.
  this->LoadVar = lowLoad ? LOAD_LOW : LOAD_HIGH;

  auto numThreads = this->environment().requestedNumThreads();

  // create a std::vector<std::shared_ptr<>> of requestenNumThreads()
  // communication pointers and add these to the threaddata
  if (ErrorDetection) {
    for (uint64_t i = 0; i < numThreads; i++) {
      auto commPtr = reinterpret_cast<uint64_t*>(ALIGNED_MALLOC(2 * sizeof(uint64_t), 64));
      assert(commPtr);
      this->ErrorCommunication.push_back(std::shared_ptr<uint64_t>(commPtr, aligned_free_deleter));
      log::debug() << "Threads " << (i + numThreads - 1) % numThreads << " and " << i << " commPtr = 0x"
                   << std::setfill('0') << std::setw(sizeof(uint64_t) * 2) << std::hex << (uint64_t)commPtr;
    }
  }

  for (uint64_t i = 0; i < numThreads; i++) {
    auto td =
        std::make_shared<LoadWorkerData>(i, this->environment(), &this->LoadVar, period, DumpRegisters, ErrorDetection);

    if (ErrorDetection) {
      // distribute pointers for error deteciton. (set threads in a ring)
      // give this thread the left pointer i and right pointer (i+1) %
      // requestedNumThreads().
      td->setErrorCommunication(this->ErrorCommunication[i], this->ErrorCommunication[(i + 1) % numThreads]);
    }

    auto dataCacheSizeIt = td->config().platformConfig().dataCacheBufferSize().begin();
    auto ramBufferSize = td->config().platformConfig().ramBufferSize();

    td->BuffersizeMem =
        (*dataCacheSizeIt + *std::next(dataCacheSizeIt, 1) + *std::next(dataCacheSizeIt, 2) + ramBufferSize) /
        td->config().thread() / sizeof(uint64_t);

    // create the thread
    std::thread t(Firestarter::loadThreadWorker, td);

    log::trace() << "Created thread #" << i << " with ID: " << t.get_id();

    if (i == 0) {
      // only show error for all worker threads except first.
      firestarter::logging::FirstWorkerThreadFilter<firestarter::logging::Record>::setFirstThread(t.get_id());
    }

    this->LoadThreads.push_back(std::make_pair(std::move(t), td));
  }

  this->signalLoadWorkers(THREAD_INIT);

  return EXIT_SUCCESS;
}

void Firestarter::signalLoadWorkers(int comm) {
  bool ack;

  // start the work
  for (auto const& thread : this->LoadThreads) {
    auto td = thread.second;

    td->Mutex.lock();
  }

  for (auto const& thread : this->LoadThreads) {
    auto td = thread.second;

    td->Comm = comm;
    td->Mutex.unlock();
  }

  for (auto const& thread : this->LoadThreads) {
    auto td = thread.second;

    do {
      td->Mutex.lock();
      ack = td->Ack;
      td->Mutex.unlock();
    } while (!ack);

    td->Mutex.lock();
    td->Ack = false;
    td->Mutex.unlock();
  }
}

void Firestarter::joinLoadWorkers() {
  // wait for threads after watchdog has requested termination
  for (auto& thread : this->LoadThreads) {
    thread.first.join();
  }
}

void Firestarter::printThreadErrorReport() {
  if (ErrorDetection) {
    auto maxSize = this->LoadThreads.size();

    std::vector<bool> errors(maxSize, false);

    for (decltype(maxSize) i = 0; i < maxSize; i++) {
      auto errorDetectionStruct = this->LoadThreads[i].second->errorDetectionStruct();

      if (errorDetectionStruct->ErrorLeft) {
        errors[(i + maxSize - 1) % maxSize] = true;
      }
      if (errorDetectionStruct->ErrorRight) {
        errors[i] = true;
      }
    }

    for (decltype(maxSize) i = 0; i < maxSize; i++) {
      if (errors[i]) {
        log::fatal() << "Data mismatch between Threads " << i << " and " << (i + 1) % maxSize
                     << ".\n       This may be caused by bit-flips in the hardware.";
      }
    }
  }
}

void Firestarter::printPerformanceReport() {
  // performance report
  uint64_t startTimestamp = 0xffffffffffffffff;
  uint64_t stopTimestamp = 0;

  uint64_t iterations = 0;

  log::debug() << "\nperformance report:\n";

  for (auto const& thread : this->LoadThreads) {
    auto td = thread.second;

    log::debug() << "Thread " << td->id() << ": " << td->Iterations
                 << " iterations, tsc_delta: " << td->StopTsc - td->StartTsc;

    if (startTimestamp > td->StartTsc) {
      startTimestamp = td->StartTsc;
    }
    if (stopTimestamp < td->StopTsc) {
      stopTimestamp = td->StopTsc;
    }

    iterations += td->Iterations;
  }

  double runtime = (double)(stopTimestamp - startTimestamp) / (double)this->environment().topology().clockrate();
  double gFlops =
      (double)this->LoadThreads.front().second->config().payload().flops() * 0.000000001 * (double)iterations / runtime;
  double bandwidth =
      (double)this->LoadThreads.front().second->config().payload().bytes() * 0.000000001 * (double)iterations / runtime;

  // insert values for ipc-estimate metric
  // if we are on linux
#if defined(linux) || defined(__linux__)
  if (Measurement) {
    for (auto const& thread : this->LoadThreads) {
      auto td = thread.second;
      ipcEstimateMetricInsert((double)td->Iterations *
                              (double)this->LoadThreads.front().second->config().payload().instructions() /
                              (double)(stopTimestamp - startTimestamp));
    }
  }
#endif

  // format runtime, gflops and bandwidth %.2f
  const char* fmt = "%.2f";
  int size;

#define FORMAT(input)                                                                                                  \
  size = std::snprintf(nullptr, 0, fmt, input);                                                                        \
  std::vector<char> input##Vector(size + 1);                                                                           \
  std::snprintf(&input##Vector[0], input##Vector.size(), fmt, input);                                                  \
  auto input##String = std::string(&input##Vector[0])

  FORMAT(runtime);
  FORMAT(gFlops);
  FORMAT(bandwidth);

#undef FORMAT

  log::debug() << "\n"
               << "total iterations: " << iterations << "\n"
               << "runtime: " << runtimeString << " seconds (" << stopTimestamp - startTimestamp << " cycles)\n"
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
    td->Mutex.lock();
    int comm = td->Comm;
    td->Mutex.unlock();

    if (comm != old) {
      old = comm;

      td->Mutex.lock();
      td->Ack = true;
      td->Mutex.unlock();
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
      td->config().payload().compilePayload(td->config().payloadSettings(), td->config().instructionCacheSize(),
                                            td->config().dataCacheBufferSize(), td->config().ramBufferSize(),
                                            td->config().thread(), td->config().lines(), td->DumpRegisters,
                                            td->ErrorDetection);

      // allocate memory
      // if we should dump some registers, we use the first part of the memory
      // for them.
      td->AddrMem =
          reinterpret_cast<uint64_t*>(ALIGNED_MALLOC((td->BuffersizeMem + td->AddrOffset) * sizeof(uint64_t), 64)) +
          td->AddrOffset;

      // exit application on error
      if (td->AddrMem - td->AddrOffset == nullptr) {
        workerLog::error() << "Could not allocate memory for CPU load thread " << td->id() << "\n";
        exit(ENOMEM);
      }

      if (td->DumpRegisters) {
        reinterpret_cast<DumpRegisterStruct*>(td->AddrMem - td->AddrOffset)->DumpVar = DumpVariable::Wait;
      }

      if (td->ErrorDetection) {
        auto errorDetectionStruct = reinterpret_cast<ErrorDetectionStruct*>(td->AddrMem - td->AddrOffset);

        std::memset(errorDetectionStruct, 0, sizeof(ErrorDetectionStruct));

        // distribute left and right communication pointers
        errorDetectionStruct->CommunicationLeft = td->CommunicationLeft.get();
        errorDetectionStruct->CommunicationRight = td->CommunicationRight.get();

        // do first touch memset 0 for the communication pointers
        std::memset((void*)errorDetectionStruct->CommunicationLeft, 0, sizeof(uint64_t) * 2);
      }

      // call init function
      td->config().payload().init(td->AddrMem, td->BuffersizeMem);

      break;
    // perform stress test
    case THREAD_WORK:
      // record threads start timestamp
      td->StartTsc = td->environment().topology().timestamp();

      // will be terminated by watchdog
      for (;;) {
        // call high load function
#ifdef ENABLE_VTRACING
        VT_USER_START("HIGH_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_BEGIN("HIGH", SCOREP_USER_REGION_TYPE_COMMON);
#endif
        td->Iterations = td->config().payload().highLoadFunction(td->AddrMem, td->AddrHigh, td->Iterations);

        // call low load function
#ifdef ENABLE_VTRACING
        VT_USER_END("HIGH_LOAD_FUNC");
        VT_USER_START("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("HIGH");
        SCOREP_USER_REGION_BY_NAME_BEGIN("LOW", SCOREP_USER_REGION_TYPE_COMMON);
#endif
        td->config().payload().lowLoadFunction(td->AddrHigh, td->Period);
#ifdef ENABLE_VTRACING
        VT_USER_END("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("LOW");
#endif

        // terminate if master signals end of run and record stop timestamp
        if (*td->AddrHigh == LOAD_STOP) {
          td->StopTsc = td->environment().topology().timestamp();

          return;
        }

        if (*td->AddrHigh == LOAD_SWITCH) {
          td->StopTsc = td->environment().topology().timestamp();

          break;
        }
      }
      break;
    case THREAD_SWITCH:
      // compile payload
      td->config().payload().compilePayload(td->config().payloadSettings(), td->config().instructionCacheSize(),
                                            td->config().dataCacheBufferSize(), td->config().ramBufferSize(),
                                            td->config().thread(), td->config().lines(), td->DumpRegisters,
                                            td->ErrorDetection);

      // call init function
      td->config().payload().init(td->AddrMem, td->BuffersizeMem);

      // save old iteration count
      td->LastIterations = td->Iterations;
      td->LastStartTsc = td->StartTsc;
      td->LastStopTsc = td->StopTsc;
      td->Iterations = 0;
      break;
    case THREAD_WAIT:
      break;
    case THREAD_STOP:
    default:
      firestarter::log::debug() << "ERR" << '\n';
      return;
    }
  }
}
