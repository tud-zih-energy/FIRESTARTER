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

#include <csignal>
#include <firestarter/Environment/X86/X86Environment.hpp>
#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>
#include <firestarter/Measurement/Metric/IPCEstimate.h>
#include <firestarter/Optimizer/Algorithm/NSGA2.hpp>
#include <firestarter/Optimizer/History.hpp>
#include <firestarter/Optimizer/Problem/CLIArgumentProblem.hpp>
#include <firestarter/WindowsCompat.hpp>

namespace firestarter {

Firestarter::Firestarter(const int Argc, const char** Argv, std::chrono::seconds const& Timeout, unsigned LoadPercent,
                         std::chrono::microseconds const& Period, unsigned RequestedNumThreads,
                         std::string const& CpuBind, bool PrintFunctionSummary, unsigned FunctionId,
                         bool ListInstructionGroups, std::string const& InstructionGroups, unsigned LineCount,
                         bool AllowUnavailablePayload, bool DumpRegisters,
                         std::chrono::seconds const& DumpRegistersTimeDelta, std::string DumpRegistersOutpath,
                         bool ErrorDetection, int Gpus, unsigned GpuMatrixSize, bool GpuUseFloat, bool GpuUseDouble,
                         bool ListMetrics, bool Measurement, std::chrono::milliseconds const& StartDelta,
                         std::chrono::milliseconds const& StopDelta,
                         std::chrono::milliseconds const& MeasurementInterval,
                         std::vector<std::string> const& MetricPaths, std::vector<std::string> const& StdinMetrics,
                         bool Optimize, std::chrono::seconds const& Preheat, std::string const& OptimizationAlgorithm,
                         std::vector<std::string> const& OptimizationMetrics,
                         std::chrono::seconds const& EvaluationDuration, unsigned Individuals,
                         std::string OptimizeOutfile, unsigned Generations, double Nsga2Cr, double Nsga2M)
    : Argc(Argc)
    , Argv(Argv)
    , Timeout(Timeout)
    , LoadPercent(LoadPercent)
    , Period(Period)
    , DumpRegisters(DumpRegisters)
    , DumpRegistersTimeDelta(DumpRegistersTimeDelta)
    , DumpRegistersOutpath(std::move(DumpRegistersOutpath))
    , ErrorDetection(ErrorDetection)
    , Gpus(Gpus)
    , GpuMatrixSize(GpuMatrixSize)
    , GpuUseFloat(GpuUseFloat)
    , GpuUseDouble(GpuUseDouble)
    , StartDelta(StartDelta)
    , StopDelta(StopDelta)
    , Measurement(Measurement)
    , Optimize(Optimize)
    , Preheat(Preheat)
    , OptimizationAlgorithm(OptimizationAlgorithm)
    , OptimizationMetrics(OptimizationMetrics)
    , EvaluationDuration(EvaluationDuration)
    , Individuals(Individuals)
    , OptimizeOutfile(std::move(OptimizeOutfile))
    , Generations(Generations)
    , Nsga2Cr(Nsga2Cr)
    , Nsga2M(Nsga2M) {
  Load = (Period * LoadPercent) / 100;
  if (LoadPercent == 100 || Load == std::chrono::microseconds::zero()) {
    this->Period = std::chrono::microseconds::zero();
  }

  if constexpr (!firestarter::OptionalFeatures.OptimizationEnabled) {
    (void)ListMetrics;
    (void)MeasurementInterval;
    (void)MetricPaths;
    (void)StdinMetrics;
  }

  if constexpr (firestarter::OptionalFeatures.IsX86) {
    Environment = std::make_unique<environment::x86::X86Environment>();
  }

  Environment->evaluateCpuAffinity(RequestedNumThreads, CpuBind);

  if constexpr (firestarter::OptionalFeatures.IsX86) {
    // Error detection uses crc32 instruction added by the SSE4.2 extension to x86
    if (ErrorDetection) {
      const auto& X86Env = *dynamic_cast<environment::x86::X86Environment*>(Environment.get());
      if (!X86Env.topology().featuresAsmjit().has(asmjit::CpuFeatures::X86::kSSE4_2)) {
        throw std::invalid_argument("Option --error-detection requires the crc32 "
                                    "instruction added with SSE_4_2.\n");
      }
    }
  }

  if (ErrorDetection && Environment->requestedNumThreads() < 2) {
    throw std::invalid_argument("Option --error-detection must run with 2 or more threads. Number of "
                                "threads is " +
                                std::to_string(Environment->requestedNumThreads()) + "\n");
  }

  Environment->evaluateFunctions();

  if (PrintFunctionSummary) {
    Environment->printFunctionSummary();
    std::exit(EXIT_SUCCESS);
  }

  Environment->selectFunction(FunctionId, AllowUnavailablePayload);

  if (ListInstructionGroups) {
    Environment->printAvailableInstructionGroups();
    std::exit(EXIT_SUCCESS);
  }

  if (!InstructionGroups.empty()) {
    Environment->selectInstructionGroups(InstructionGroups);
  }

  if (LineCount != 0) {
    Environment->setLineCount(LineCount);
  }

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    if (Measurement || ListMetrics || Optimize) {
      MeasurementWorker = std::make_shared<measurement::MeasurementWorker>(
          MeasurementInterval, Environment->requestedNumThreads(), MetricPaths, StdinMetrics);

      if (ListMetrics) {
        log::info() << MeasurementWorker->availableMetrics();
        std::exit(EXIT_SUCCESS);
      }

      // init all metrics
      auto All = MeasurementWorker->metricNames();
      auto Initialized = MeasurementWorker->initMetrics(All);

      if (Initialized.empty()) {
        std::invalid_argument("No metrics initialized");
      }

      // check if selected metrics are initialized
      for (auto const& OptimizationMetric : OptimizationMetrics) {
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

    if (Optimize) {
      auto ApplySettings = std::bind(
          [this](std::vector<std::pair<std::string, unsigned>> const& Setting) {
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
              ipcEstimateMetricInsert(
                  static_cast<double>(Td->LastRun.Iterations) *
                  static_cast<double>(LoadThreads.front().second->config().payload().instructions()) /
                  static_cast<double>(StopTimestamp - StartTimestamp));
            }

            auto End = Clock::now();

            log::trace() << "Switching payload took "
                         << std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count() << "ms";
          },
          std::placeholders::_1);

      auto Prob = std::make_shared<firestarter::optimizer::problem::CLIArgumentProblem>(
          std::move(ApplySettings), MeasurementWorker, OptimizationMetrics, EvaluationDuration, StartDelta, StopDelta,
          Environment->selectedConfig().payloadItems());

      Population = firestarter::optimizer::Population(std::move(Prob));

      if (OptimizationAlgorithm == "NSGA2") {
        Algorithm = std::make_unique<firestarter::optimizer::algorithm::NSGA2>(Generations, Nsga2Cr, Nsga2M);
      } else {
        throw std::invalid_argument("Algorithm " + OptimizationAlgorithm + " unknown.");
      }

      Algorithm->checkPopulation(static_cast<firestarter::optimizer::Population const&>(Population), Individuals);
    }
  }

  Environment->printSelectedCodePathSummary();

  log::info() << Environment->topology();

  // setup thread with either high or low load configured at the start
  // low loads has to know the length of the period
  initLoadWorkers((LoadPercent == 0), Period);

  // add some signal handler for aborting FIRESTARTER
  if constexpr (!firestarter::OptionalFeatures.IsWin32) {
    std::signal(SIGALRM, Firestarter::sigalrmHandler);
  }

  std::signal(SIGTERM, Firestarter::sigtermHandler);
  std::signal(SIGINT, Firestarter::sigtermHandler);
}

void Firestarter::mainThread() {
  Environment->printThreadSummary();

  Cuda = std::make_unique<cuda::Cuda>(LoadVar, GpuUseFloat, GpuUseDouble, GpuMatrixSize, Gpus);
  Oneapi = std::make_unique<oneapi::OneAPI>(LoadVar, GpuUseFloat, GpuUseDouble, GpuMatrixSize, Gpus);

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    // if measurement is enabled, start it here
    if (Measurement) {
      MeasurementWorker->startMeasurement();
    }
  }

  signalWork();

  if constexpr (firestarter::OptionalFeatures.DumpRegisterEnabled) {
    if (DumpRegisters) {
      initDumpRegisterWorker(DumpRegistersTimeDelta, DumpRegistersOutpath);
    }
  }

  // worker thread for load control
  watchdogWorker(Period, Load, Timeout);

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    // check if optimization is selected
    if (Optimize) {
      auto StartTime = optimizer::History::getTime();

      Firestarter::Optimizer = std::make_unique<optimizer::OptimizerWorker>(
          std::move(Algorithm), Population, OptimizationAlgorithm, Individuals, Preheat);

      // wait here until optimizer thread terminates
      Firestarter::Optimizer->join();

      auto PayloadItems = Environment->selectedConfig().payloadItems();

      firestarter::optimizer::History::save(OptimizeOutfile, StartTime, PayloadItems, Argc, Argv);

      // print the best 20 according to each metric
      firestarter::optimizer::History::printBest(OptimizationMetrics, PayloadItems);

      // stop all the load threads
      std::raise(SIGTERM);
    }
  }

  // wait for watchdog to timeout or until user terminates
  joinLoadWorkers();
  if constexpr (firestarter::OptionalFeatures.DumpRegisterEnabled) {
    if (DumpRegisters) {
      joinDumpRegisterWorker();
    }
  }

  if (!Optimize) {
    printPerformanceReport();
  }

  if constexpr (firestarter::OptionalFeatures.OptimizationEnabled) {
    // if measurment is enabled, stop it here
    if (Measurement) {
      // TODO: clear this up
      log::info() << "metric,num_timepoints,duration_ms,average,stddev";
      for (auto const& [name, sum] : MeasurementWorker->getValues(StartDelta, StopDelta)) {
        log::info() << std::quoted(name) << "," << sum.NumTimepoints << "," << sum.Duration.count() << ","
                    << sum.Average << "," << sum.Stddev;
      }
    }
  }

  if (ErrorDetection) {
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