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
#include <string>
#include <vector>

namespace firestarter {

struct Config {
  const char** Argv;

  std::chrono::seconds Timeout{};
  std::chrono::microseconds Period{};
  std::chrono::microseconds Load{};

  std::chrono::seconds DumpRegistersTimeDelta = std::chrono::seconds(0);
  std::chrono::milliseconds StartDelta = std::chrono::milliseconds(0);
  std::chrono::milliseconds StopDelta = std::chrono::milliseconds(0);
  std::chrono::milliseconds MeasurementInterval = std::chrono::milliseconds(0);
  std::chrono::seconds Preheat{};
  std::chrono::seconds EvaluationDuration{};

  double Nsga2Cr;
  double Nsga2M;

  std::vector<std::string> StdinMetrics;
  std::vector<std::string> MetricPaths;
  std::vector<std::string> OptimizationMetrics;

  std::string CpuBind;
  std::string InstructionGroups;
  std::string DumpRegistersOutpath;
  std::string OptimizationAlgorithm;
  std::string OptimizeOutfile;

  int Argc;
  unsigned RequestedNumThreads;
  unsigned FunctionId;
  unsigned LineCount = 0;
  int Gpus = 0;
  unsigned GpuMatrixSize = 0;
  unsigned Individuals;
  unsigned Generations;

  bool PrintFunctionSummary;
  bool ListInstructionGroups;
  bool AllowUnavailablePayload = false;
  bool DumpRegisters = false;
  bool ErrorDetection = false;
  bool GpuUseFloat = false;
  bool GpuUseDouble = false;
  bool ListMetrics = false;
  bool Measurement = false;
  bool Optimize = false;

  Config() = delete;

  Config(int Argc, const char** Argv);
};

} // namespace firestarter