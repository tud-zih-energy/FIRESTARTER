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

#include "firestarter/Firestarter.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/Measurement/Metric/IPCEstimate.hpp"
#include "firestarter/Optimizer/Algorithm/NSGA2.hpp"
#include "firestarter/Optimizer/History.hpp"
#include "firestarter/Optimizer/Problem/CLIArgumentProblem.hpp"
#include "firestarter/X86/X86FunctionSelection.hpp"

#include <csignal>
#include <cstdlib>
#include <memory>

namespace firestarter {

Firestarter::Firestarter(Config&& ProvidedConfig)
    : Cfg(std::move(ProvidedConfig)) {
  std::unique_ptr<FunctionSelection> FunctionSelectionPtr;

  if constexpr (firestarter::OptionalFeatures.IsX86) {
    ProcessorInfos = std::make_shared<x86::X86ProcessorInformation>();
    FunctionSelectionPtr = std::make_unique<x86::X86FunctionSelection>();
  }

  const auto Affinity =
      ThreadAffinity::fromCommandLine(Topology.hardwareThreadsInfo(), Cfg.RequestedNumThreads, Cfg.CpuBinding);

  if constexpr (firestarter::OptionalFeatures.IsX86) {
    // Error detection uses crc32 instruction added by the SSE4.2 extension to x86
    if (Cfg.ErrorDetection) {
      const auto& X86ProcessorInfos = *dynamic_cast<x86::X86ProcessorInformation*>(ProcessorInfos.get());
      if (!X86ProcessorInfos.featuresAsmjit().has(asmjit::CpuFeatures::X86::kSSE4_2)) {
        throw std::invalid_argument("Option --error-detection requires the crc32 "
                                    "instruction added with SSE_4_2.\n");
      }
    }
  }

  if (Cfg.ErrorDetection && Affinity.RequestedNumThreads < 2) {
    throw std::invalid_argument("Option --error-detection must run with 2 or more threads. Number of "
                                "threads is " +
                                std::to_string(Affinity.RequestedNumThreads) + "\n");
  }

  if (Cfg.PrintFunctionSummary) {
    FunctionSelectionPtr->printFunctionSummary(*ProcessorInfos, /*ForceYes=*/false);
    safeExit(EXIT_SUCCESS);
  }

  FunctionPtr =
      FunctionSelectionPtr->selectFunction(Cfg.FunctionId, *ProcessorInfos, Topology, Cfg.AllowUnavailablePayload);

  if (Cfg.ListInstructionGroups) {
    FunctionPtr->payload()->printAvailableInstructionGroups();

    safeExit(EXIT_SUCCESS);
  }
  if (Cfg.Groups) {
    FunctionPtr->selectInstructionGroups(*Cfg.Groups);
    log::info() << "  Running custom instruction group: " << *Cfg.Groups;
  }

  if (Cfg.LineCount) {
    FunctionPtr->settings().setLineCount(*Cfg.LineCount);
  }

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    if (Cfg.Measurement || Cfg.ListMetrics || Cfg.Optimize) {
      MeasurementWorker = std::make_shared<measurement::MeasurementWorker>(
          Cfg.MeasurementInterval, Affinity.RequestedNumThreads, Cfg.MetricPaths, Cfg.StdinMetrics);

      if (Cfg.ListMetrics) {
        log::info() << MeasurementWorker->availableMetrics();
        safeExit(EXIT_SUCCESS);
      }

      // init all metrics
      auto All = MeasurementWorker->metricNames();
      auto Initialized = MeasurementWorker->initMetrics(All);

      if (Initialized.empty()) {
        std::invalid_argument("No metrics initialized");
      }

      // check if selected metrics are initialized
      for (auto const& OptimizationMetric : Cfg.OptimizationMetrics) {
        auto NameEqual = [OptimizationMetric](auto const& Name) {
          auto InvertedName = "-" + Name;
          return Name == OptimizationMetric || InvertedName == OptimizationMetric;
        };
        // metric name is not found
        if (std::find_if(All.begin(), All.end(), NameEqual) == All.end()) {
          std::invalid_argument("Metric \"" + OptimizationMetric + "\" does not exist.");
        }
        // metric has not initialized properly
        if (std::find_if(Initialized.begin(), Initialized.end(), NameEqual) == Initialized.end()) {
          std::invalid_argument("Metric \"" + OptimizationMetric + "\" failed to initialize.");
        }
      }
    }

    if (Cfg.Optimize) {
      auto ApplySettings = [this](InstructionGroups const& Setting) {
        using Clock = std::chrono::high_resolution_clock;
        auto Start = Clock::now();

        signalSwitch(Setting);

        LoadVar = LoadThreadWorkType::LoadHigh;

        signalWork();

        uint64_t StartTimestamp = (std::numeric_limits<uint64_t>::max)();
        uint64_t StopTimestamp = 0;

        for (auto const& Thread : LoadThreads) {
          auto Td = Thread.second;

          StartTimestamp = std::min<uint64_t>(StartTimestamp, Td->LastRun.StartTsc);
          StopTimestamp = std::max<uint64_t>(StopTimestamp, Td->LastRun.StopTsc);
        }

        for (auto const& Thread : LoadThreads) {
          auto Td = Thread.second;
          IpcEstimateMetricData::insertValue(
              static_cast<double>(Td->LastRun.Iterations) *
              static_cast<double>(LoadThreads.front().second->CompiledPayloadPtr->stats().Instructions) /
              static_cast<double>(StopTimestamp - StartTimestamp));
        }

        auto End = Clock::now();

        log::trace() << "Switching payload took "
                     << std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count() << "ms";
      };

      auto Prob = std::make_shared<firestarter::optimizer::problem::CLIArgumentProblem>(
          std::move(ApplySettings), MeasurementWorker, Cfg.OptimizationMetrics, Cfg.EvaluationDuration, Cfg.StartDelta,
          Cfg.StopDelta, FunctionPtr->settings().groups().intructions());

      Population = std::make_unique<firestarter::optimizer::Population>(std::move(Prob));

      if (Cfg.OptimizationAlgorithm == "NSGA2") {
        Algorithm =
            std::make_unique<firestarter::optimizer::algorithm::NSGA2>(Cfg.Generations, Cfg.Nsga2Cr, Cfg.Nsga2M);
      } else {
        throw std::invalid_argument("Algorithm " + Cfg.OptimizationAlgorithm + " unknown.");
      }

      Algorithm->check(Population->problem(), Cfg.Individuals);
    }
  }

  FunctionPtr->printCodePathSummary();

  Topology.printSystemSummary();
  log::info();
  ProcessorInfos->print();
  Topology.printCacheSummary();

  Affinity.printThreadSummary(Topology);

  // setup thread with either high or low load configured at the start
  // low loads has to know the length of the period
  initLoadWorkers(Affinity);

  // add some signal handler for aborting FIRESTARTER
  if constexpr (!firestarter::OptionalFeatures.IsWin32) {
    (void)std::signal(SIGALRM, Firestarter::sigalrmHandler);
  }

  (void)std::signal(SIGTERM, Firestarter::sigtermHandler);
  (void)std::signal(SIGINT, Firestarter::sigtermHandler);
}

void Firestarter::mainThread() {
  Cuda = std::make_unique<cuda::Cuda>(LoadVar, Cfg.GpuUseFloat, Cfg.GpuUseDouble, Cfg.GpuMatrixSize, Cfg.Gpus);
  Oneapi = std::make_unique<oneapi::OneAPI>(LoadVar, Cfg.GpuUseFloat, Cfg.GpuUseDouble, Cfg.GpuMatrixSize, Cfg.Gpus);

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    // if measurement is enabled, start it here
    if (Cfg.Measurement) {
      MeasurementWorker->startMeasurement();
    }
  }

  signalWork();

  if constexpr (firestarter::OptionalFeatures.DumpRegisterEnabled) {
    if (Cfg.DumpRegisters) {
      initDumpRegisterWorker();
    }
  }

  // worker thread for load control
  watchdogWorker(Cfg.Period, Cfg.Load, Cfg.Timeout);

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    // check if optimization is selected
    if (Cfg.Optimize) {
      auto StartTime = optimizer::History::getTime();

      Firestarter::Optimizer = std::make_unique<optimizer::OptimizerWorker>(std::move(Algorithm), std::move(Population),
                                                                            Cfg.Individuals, Cfg.Preheat);

      // wait here until optimizer thread terminates
      Firestarter::Optimizer->join();
      Firestarter::Optimizer.reset();

      auto PayloadItems = FunctionPtr->settings().groups().intructions();

      firestarter::optimizer::History::save(Cfg.OptimizeOutfile, StartTime, PayloadItems, Cfg.Argc, Cfg.Argv);

      // print the best 20 according to each metric
      firestarter::optimizer::History::printBest(Cfg.OptimizationMetrics, PayloadItems);

      // stop all the load threads
      (void)std::raise(SIGTERM);
    }
  }

  // wait for watchdog to timeout or until user terminates
  joinLoadWorkers();
  if constexpr (firestarter::OptionalFeatures.DumpRegisterEnabled) {
    if (Cfg.DumpRegisters) {
      joinDumpRegisterWorker();
    }
  }

  if (!Cfg.Optimize) {
    printPerformanceReport();
  }

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    // if measurment is enabled, stop it here
    if (Cfg.Measurement) {
      // TODO(Issue #77): clear this up
      log::info() << "metric,num_timepoints,duration_ms,average,stddev";
      for (auto const& [name, sum] : MeasurementWorker->getValues(Cfg.StartDelta, Cfg.StopDelta)) {
        log::info() << std::quoted(name) << "," << sum.NumTimepoints << "," << sum.Duration.count() << ","
                    << sum.Average << "," << sum.Stddev;
      }
    }
  }

  if (Cfg.ErrorDetection) {
    printThreadErrorReport();
  }
}

void Firestarter::setLoad(LoadThreadWorkType Value) {
  // signal load change to workers
  Firestarter::LoadVar = Value;
  if constexpr (firestarter::OptionalFeatures.IsX86) {
    if constexpr (firestarter::OptionalFeatures.IsMsc) {
      _mm_mfence();
    } else {
      __asm__ __volatile__("mfence;");
    }
  }
}

void Firestarter::sigalrmHandler(int Signum) { (void)Signum; }

void Firestarter::sigtermHandler(int Signum) {
  (void)Signum;

  Firestarter::setLoad(LoadThreadWorkType::LoadStop);
  // exit loop
  // used in case of 0 < load < 100
  // or interrupt sleep for timeout
  {
    const std::lock_guard<std::mutex> Lk(Firestarter::WatchdogTerminateMutex);
    Firestarter::WatchdogTerminate = true;
  }
  Firestarter::WatchdogTerminateAlert.notify_all();

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    // if we have optimization running stop it
    if (Firestarter::Optimizer) {
      Firestarter::Optimizer->kill();
    }
  }
}

} // namespace firestarter