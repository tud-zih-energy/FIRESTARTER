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

#include "firestarter/AlignedAlloc.hpp"
#include "firestarter/Constants.hpp"
#include "firestarter/ErrorDetectionStruct.hpp"
#include "firestarter/Firestarter.hpp"
#include "firestarter/LoadWorkerData.hpp"
#include "firestarter/Logging/Log.hpp"

#if defined(linux) || defined(__linux__)
#include "firestarter/Measurement/Metric/IPCEstimate.hpp"
#endif

#ifdef ENABLE_VTRACING
#include <vt_user.h>
#endif
#ifdef ENABLE_SCOREP
#include <SCOREP_User.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

namespace firestarter {

void Firestarter::initLoadWorkers(const ThreadAffinity& Affinity) {
  // Bind this thread to the first available CPU.
  Topology.bindCallerToOsIndex(*Affinity.CpuBind.cbegin());

  // setup load variable to execute low or high load once the threads switch to
  // work.
  LoadVar = Cfg.Load == std::chrono::microseconds::zero() ? LoadThreadWorkType::LoadLow : LoadThreadWorkType::LoadHigh;

  // create a std::vector<std::shared_ptr<>> of requestenNumThreads()
  // communication pointers and add these to the threaddata
  if (Cfg.ErrorDetection) {
    for (uint64_t I = 0; I < Affinity.RequestedNumThreads; I++) {
      auto* CommPtr = static_cast<uint64_t*>(AlignedAlloc::malloc(2 * sizeof(uint64_t)));
      assert(CommPtr);
      ErrorCommunication.emplace_back(std::shared_ptr<uint64_t>(CommPtr, AlignedAlloc::free));
      log::debug() << "Threads " << (I + Affinity.RequestedNumThreads - 1) % Affinity.RequestedNumThreads << " and "
                   << I << " commPtr = 0x" << std::setfill('0') << std::setw(sizeof(uint64_t) * 2)
                   << std::hex
                   // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                   << reinterpret_cast<uint64_t>(CommPtr);
    }
  }

  uint64_t I = 0;
  for (const auto& OsIndex : Affinity.CpuBind) {
    auto Td = std::make_shared<LoadWorkerData>(I, OsIndex, std::cref(ProcessorInfos), std::cref(FunctionPtr),
                                               std::cref(Topology), std::ref(LoadVar), Cfg.Period, Cfg.DumpRegisters,
                                               Cfg.ErrorDetection);

    if (Cfg.ErrorDetection) {
      // distribute pointers for error deteciton. (set threads in a ring)
      // give this thread the left pointer i and right pointer (i+1) %
      // requestedNumThreads().
      Td->setErrorCommunication(ErrorCommunication[I], ErrorCommunication[(I + 1) % Affinity.RequestedNumThreads]);
    }

    Td->BuffersizeMem = Td->config().settings().totalBufferSizePerThread() / sizeof(uint64_t);

    // create the thread
    std::thread T(Firestarter::loadThreadWorker, Td);

    log::trace() << "Created thread #" << I << " with ID: " << T.get_id();

    if (I == 0) {
      // only show error for all worker threads except first.
      firestarter::logging::FirstWorkerThreadFilter<firestarter::logging::record>::setFirstThread(T.get_id());
    }

    LoadThreads.emplace_back(std::move(T), Td);

    I++;
  }

  signalLoadWorkers(LoadThreadState::ThreadInit);
}

void Firestarter::signalLoadWorkers(const LoadThreadState State, void (*Function)()) {
  // aquire the lock on all threads
  for (auto const& Thread : LoadThreads) {
    auto Td = Thread.second;

    Td->Communication.Mutex.lock();
  }

  // switch the state on all threads
  for (auto const& Thread : LoadThreads) {
    auto Td = Thread.second;

    Td->Communication.State = State;
    Td->Communication.Mutex.unlock();
  }

  // Execute a function after the state in the threads has been updated. This may be required to terminate an inner
  // loop.
  if (Function) {
    Function();
  }

  // wait for all threads to finish
  for (auto const& Thread : LoadThreads) {
    auto Td = Thread.second;

    // Wait until we receive the acknowledge
    for (bool Ack = false; !Ack;) {
      Td->Communication.Mutex.lock();
      Ack = Td->Communication.Ack;
      Td->Communication.Mutex.unlock();
    }

    Td->Communication.Mutex.lock();
    Td->Communication.Ack = false;
    Td->Communication.Mutex.unlock();
  }
}

void Firestarter::joinLoadWorkers() {
  // wait for threads after watchdog has requested termination
  for (auto& Thread : LoadThreads) {
    Thread.first.join();
  }
}

void Firestarter::printThreadErrorReport() {
  if (Cfg.ErrorDetection) {
    auto MaxSize = LoadThreads.size();

    std::vector<bool> Errors(MaxSize, false);

    for (decltype(MaxSize) I = 0; I < MaxSize; I++) {
      const auto& ErrorDetectionStructPtr = LoadThreads[I].second->errorDetectionStruct();

      if (ErrorDetectionStructPtr.Left.Error) {
        Errors[(I + MaxSize - 1) % MaxSize] = true;
      }
      if (ErrorDetectionStructPtr.Right.Error) {
        Errors[I] = true;
      }
    }

    for (decltype(MaxSize) I = 0; I < MaxSize; I++) {
      if (Errors[I]) {
        log::fatal() << "Data mismatch between Threads " << I << " and " << (I + 1) % MaxSize
                     << ".\n       This may be caused by bit-flips in the hardware.";
      }
    }
  }
}

void Firestarter::printPerformanceReport() {
  // performance report
  uint64_t StartTimestamp = (std::numeric_limits<uint64_t>::max)();
  uint64_t StopTimestamp = 0;

  uint64_t Iterations = 0;

  log::debug() << "\nperformance report:\n";

  for (auto const& Thread : LoadThreads) {
    auto Td = Thread.second;

    log::debug() << "Thread " << Td->id() << ": " << Td->LastRun.Iterations
                 << " iterations, tsc_delta: " << Td->LastRun.StopTsc - Td->LastRun.StartTsc;

    StartTimestamp = (std::min)(StartTimestamp, Td->LastRun.StartTsc.load());
    StopTimestamp = (std::max)(StopTimestamp, Td->LastRun.StopTsc.load());

    Iterations += Td->LastRun.Iterations.load();
  }

  double const Runtime =
      static_cast<double>(StopTimestamp - StartTimestamp) / static_cast<double>(ProcessorInfos->clockrate());
  double const GFlops = static_cast<double>(LoadThreads.front().second->CompiledPayloadPtr->stats().Flops) *
                        0.000000001 * static_cast<double>(Iterations) / Runtime;
  double const Bandwidth = static_cast<double>(LoadThreads.front().second->CompiledPayloadPtr->stats().Bytes) *
                           0.000000001 * static_cast<double>(Iterations) / Runtime;

  // insert values for ipc-estimate metric
  // if we are on linux
#if defined(linux) || defined(__linux__)
  if (Cfg.Measurement) {
    for (auto const& Thread : LoadThreads) {
      auto Td = Thread.second;
      IpcEstimateMetricData::insertValue(
          static_cast<double>(Td->LastRun.Iterations) *
          static_cast<double>(LoadThreads.front().second->CompiledPayloadPtr->stats().Instructions) /
          static_cast<double>(StopTimestamp - StartTimestamp));
    }
  }
#endif

  // format runtime, gflops and bandwidth with two decimal places
  const auto FormatString = [](double Value) -> std::string {
    std::stringstream Ss;
    Ss << std::fixed << std::setprecision(2) << Value;
    return Ss.str();
  };

  const auto PrintGpuFlops = [&Runtime, &FormatString](auto& GpuPtr) -> void {
    if (!GpuPtr) {
      return;
    }

    auto SingleFlops = static_cast<double>(GpuPtr->executedFlop().SingleFlop.load()) * 0.000000001 / Runtime;
    auto DoubleFlops = static_cast<double>(GpuPtr->executedFlop().DoubleFlop.load()) * 0.000000001 / Runtime;

    if (SingleFlops > 0) {
      log::debug() << "\n"
                   << "estimated floating point performance (GPU)**: " << FormatString(SingleFlops)
                   << " GFLOPS (single)";
    }

    if (DoubleFlops > 0) {
      log::debug() << "\n"
                   << "estimated floating point performance (GPU)**: " << FormatString(DoubleFlops)
                   << " GFLOPS (double)";
    }
  };

  log::debug() << "\n"
               << "total iterations: " << Iterations << "\n"
               << "runtime: " << FormatString(Runtime) << " seconds (" << StopTimestamp - StartTimestamp << " cycles)\n"
               << "\n"
               << "estimated floating point performance (CPU): " << FormatString(GFlops) << " GFLOPS\n"
               << "estimated memory bandwidth (CPU)*: " << FormatString(Bandwidth) << " GB/s";

  PrintGpuFlops(Cuda);
  PrintGpuFlops(Oneapi);

  log::debug() << "\n"
               << "* this estimate is highly unreliable if --function is used in order "
                  "to "
                  "select\n"
               << "  a function that is not optimized for your architecture, or if "
                  "FIRESTARTER is\n"
               << "  executed on an unsupported architecture!";

  if (Cuda || Oneapi) {
    log::debug()
        << "** this estimate is based on the assumption that no algorithmically optimized version\n"
        << "    of the called algorithm has been implemented by the vendor. It also might not be not accurate\n"
        << "    for short runs of FIRESTARTER";
  }
}

void Firestarter::loadThreadWorker(const std::shared_ptr<LoadWorkerData>& Td) {

  auto OldState = LoadThreadState::ThreadWait;

#if defined(linux) || defined(__linux__)
  pthread_setname_np(pthread_self(), "LoadWorker");
#endif

  for (;;) {
    Td->Communication.Mutex.lock();
    auto CurState = Td->Communication.State;
    Td->Communication.Mutex.unlock();

    if (CurState != OldState) {
      OldState = CurState;

      Td->Communication.Mutex.lock();
      Td->Communication.Ack = true;
      Td->Communication.Mutex.unlock();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      continue;
    }

    switch (CurState) {
    // allocate and initialize memory
    case LoadThreadState::ThreadInit:
      // set affinity
      Td->Topology.bindCallerToOsIndex(Td->OsIndex);

      // compile payload
      Td->CompiledPayloadPtr = Td->config().payload()->compilePayload(Td->config().settings(), Td->DumpRegisters,
                                                                      Td->ErrorDetection, /*PrintAssembler=*/false);

      // allocate memory
      // if we should dump some registers, we use the first part of the memory
      // for them.
      Td->Memory = LoadWorkerMemory::allocate(Td->BuffersizeMem * sizeof(uint64_t));

      // exit application on error
      if (Td->Memory == nullptr) {
        workerLog::error() << "Could not allocate memory for CPU load thread " << Td->id() << "\n";
      }

      if (Td->DumpRegisters) {
        Td->dumpRegisterStruct().DumpVar = DumpVariable::Wait;
      }

      if (Td->ErrorDetection) {
        auto& ErrorDetectionStructRef = Td->errorDetectionStruct();

        std::memset(&ErrorDetectionStructRef, 0, sizeof(ErrorDetectionStruct));

        // distribute left and right communication pointers
        ErrorDetectionStructRef.Left.Communication = Td->CommunicationLeft.get();
        ErrorDetectionStructRef.Right.Communication = Td->CommunicationRight.get();

        // do first touch memset 0 for the communication pointers
        std::memset(static_cast<void*>(ErrorDetectionStructRef.Left.Communication), 0, sizeof(uint64_t) * 2);
      }

      // call init function
      Td->CompiledPayloadPtr->init(Td->Memory->getMemoryAddress(), Td->BuffersizeMem);

      break;
    // perform stress test
    case LoadThreadState::ThreadWork:
      Td->CurrentRun.Iterations = 0;
      // record threads start timestamp
      Td->CurrentRun.StartTsc = Td->ProcessorInfos->timestamp();

      // will be terminated by watchdog
      for (;;) {
        // call high load function
#ifdef ENABLE_VTRACING
        VT_USER_START("HIGH_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_BEGIN("HIGH", SCOREP_USER_REGION_TYPE_COMMON);
#endif
        Td->CurrentRun.Iterations = Td->CompiledPayloadPtr->highLoadFunction(Td->Memory->getMemoryAddress(),
                                                                             Td->LoadVar, Td->CurrentRun.Iterations);

        // call low load function
#ifdef ENABLE_VTRACING
        VT_USER_END("HIGH_LOAD_FUNC");
        VT_USER_START("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("HIGH");
        SCOREP_USER_REGION_BY_NAME_BEGIN("LOW", SCOREP_USER_REGION_TYPE_COMMON);
#endif
        Td->CompiledPayloadPtr->lowLoadFunction(Td->LoadVar, Td->Period);
#ifdef ENABLE_VTRACING
        VT_USER_END("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("LOW");
#endif

        // terminate if master signals end of run and record stop timestamp
        if (Td->LoadVar == LoadThreadWorkType::LoadStop) {
          Td->CurrentRun.StopTsc = Td->ProcessorInfos->timestamp();
          Td->LastRun = Td->CurrentRun;

          return;
        }

        if (Td->LoadVar == LoadThreadWorkType::LoadSwitch) {
          Td->CurrentRun.StopTsc = Td->ProcessorInfos->timestamp();
          Td->LastRun = Td->CurrentRun;

          break;
        }
      }
      break;
    case LoadThreadState::ThreadSwitch:
      // compile payload
      Td->CompiledPayloadPtr = Td->config().payload()->compilePayload(Td->config().settings(), Td->DumpRegisters,
                                                                      Td->ErrorDetection, /*PrintAssembler=*/false);

      // call init function
      Td->CompiledPayloadPtr->init(Td->Memory->getMemoryAddress(), Td->BuffersizeMem);
      break;
    case LoadThreadState::ThreadWait:
      break;
    }
  }
}

} // namespace firestarter