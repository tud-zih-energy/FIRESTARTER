/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
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

#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace firestarter {

/// This struct contains the parsed config from the command line for Firestarter.
struct Config {
  /// The argument vector from the command line.
  const char** Argv;

  /// The timeout after which firestarter terminates. This is available in combination with optimization.
  std::chrono::seconds Timeout{};
  /// The period after with which the low/high load routine is switched.
  std::chrono::microseconds Period{};
  /// The load in the range of 0 < Load <= Period, which controls how long of the period the high-load loop runs.
  std::chrono::microseconds Load{};

  /// The interval every which the register will be dumped to the file.
  std::chrono::seconds DumpRegistersTimeDelta = std::chrono::seconds(0);
  /// The time to skip from the measurement start
  std::chrono::milliseconds StartDelta = std::chrono::milliseconds(0);
  /// The time to skip from the measurement stop
  std::chrono::milliseconds StopDelta = std::chrono::milliseconds(0);
  /// Metric values will be polled by the MeasurementInterval.
  std::chrono::milliseconds MeasurementInterval = std::chrono::milliseconds(0);
  /// The time how long the processor will be preheated before starting a measurement or optimization.
  std::chrono::seconds Preheat{};
  /// The time how long a measurement should take.
  std::chrono::seconds EvaluationDuration{};

  /// The crossover probability used in the NSGA2 optimization algorithm.
  double Nsga2Cr;
  /// The mutation probability used in the NSGA2 optimization algorithm.
  double Nsga2M;

  /// The name of the metrics that are read from stdin.
  std::vector<std::string> StdinMetrics;
  /// The paths to the metrics that are loaded using shared libraries.
  std::vector<std::string> MetricPaths;
  /// The list of metrics that are used for maximization. If a metric is prefixed with '-' it will be minimized.
  std::vector<std::string> OptimizationMetrics;

  /// The optional cpu bind that allow pinning to specific cpus.
  std::optional<std::set<uint64_t>> CpuBinding;
  /// The optional selected instruction groups. If this is empty the default will be choosen.
  std::string InstructionGroups;
  /// The file where the dump register feature will safe its output to.
  std::string DumpRegistersOutpath;
  /// The name of the optimization algorithm.
  std::string OptimizationAlgorithm;
  /// The file where the data saved during optimization is saved.
  std::string OptimizeOutfile;

  /// The argument count from the command line.
  int Argc;
  /// The requested number of threads firestarter should run with. 0 means all threads.
  std::optional<unsigned> RequestedNumThreads;
  /// The selected function id. 0 means automatic selection.
  unsigned FunctionId;
  /// The line count of the payload. 0 means default.
  unsigned LineCount = 0;
  /// The number of gpus firestarter should stress. Default is -1 means all gpus.
  int Gpus = 0;
  /// The matrix size which should be used. 0 means automatic detections.
  unsigned GpuMatrixSize = 0;
  /// The number of individuals that should be used for the optimization.
  unsigned Individuals;
  /// The number of generations that should be used for the optimization.
  unsigned Generations;

  /// If the function summary should be printed.
  bool PrintFunctionSummary;
  /// If the available instruction groups for a function should be printed.
  bool ListInstructionGroups;
  /// Allow payloads that are not supported on the current processor.
  bool AllowUnavailablePayload = false;
  /// Is the dump registers debug feature enabled?
  bool DumpRegisters = false;
  /// Is the error detection feature enabled?
  bool ErrorDetection = false;
  /// Should the GPUs use floating point precision? If neither GpuUseFloat or GpuUseDouble is set, precision will be
  /// choosen automatically.
  bool GpuUseFloat = false;
  /// Should the GPUs use double point precision? If neither GpuUseFloat or GpuUseDouble is set, precision will be
  /// choosen automatically.
  bool GpuUseDouble = false;
  /// Should we print all available metrics.
  bool ListMetrics = false;
  /// Do we perform an measurement.
  bool Measurement = false;
  /// Do we perform optimization.
  bool Optimize = false;

  Config() = delete;

  /// Parser the config from the command line argumens.
  Config(int Argc, const char** Argv);
};

} // namespace firestarter