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

#pragma once

#include "firestarter/Config/InstructionGroups.hpp"
#include "firestarter/Measurement/MeasurementWorker.hpp"
#include "firestarter/Optimizer/Problem.hpp"

#include <cassert>
#include <cmath>
#include <functional>
#include <thread>
#include <tuple>
#include <utility>

namespace firestarter::optimizer::problem {

/// This class models the problem of optimizing firestarter on the fly. The evaluation of metrics is done by switching
/// the settings of the high load routine and measuring the metric in the specified runtime.
class CLIArgumentProblem final : public firestarter::optimizer::Problem {
private:
  /// The function which takes instruction groups and switches the payload in the high load function to the supplied
  /// ones.
  std::function<void(const firestarter::InstructionGroups&)> ChangePayloadFunction;
  /// The shared pointer to the measurement infrastructure which will be used to get metric values.
  std::shared_ptr<firestarter::measurement::MeasurementWorker> MeasurementWorker;
  /// The metrics that are used in the optimization. They may have a dash at the start to allow them to be changed from
  /// maximization to minimization.
  std::set<MetricName> OptimizationMetrics;
  /// The duration of the measurement.
  std::chrono::seconds Timeout;
  /// The time to skip from the measurement start
  std::chrono::milliseconds StartDelta;
  /// The time to skip from the measurement stop
  std::chrono::milliseconds StopDelta;
  /// The vector of instruction that is used in the optimization for the payload.
  std::vector<std::string> Instructions;

public:
  /// Constructor for the problem of optimizing firestarter on the fly.
  /// \arg ChangePayloadFunction The function which takes instruction groups and switches the payload in the high load
  /// function to the supplied ones.
  /// \arg MeasurementWorker The shared pointer to the measurement infrastructure which will be used to get metric
  /// values
  /// \arg Metrics The metrics that are used in the optimization. They may have a dash at the start to allow them to be
  /// changed from maximization to minimization.
  /// \arg Timeout The duration of the measurement.
  /// \arg StartDelta The time to skip from the measurement start
  /// \arg StopDelta The time to skip from the measurement stop
  /// \arg Instructions The vector of instruction that is used in the optimization for the payload.
  CLIArgumentProblem(std::function<void(const firestarter::InstructionGroups&)>&& ChangePayloadFunction,
                     std::shared_ptr<firestarter::measurement::MeasurementWorker> MeasurementWorker,
                     std::set<MetricName> const& Metrics, std::chrono::seconds Timeout,
                     std::chrono::milliseconds StartDelta, std::chrono::milliseconds StopDelta,
                     std::vector<std::string> Instructions)
      : ChangePayloadFunction(std::move(ChangePayloadFunction))
      , MeasurementWorker(std::move(MeasurementWorker))
      , OptimizationMetrics(Metrics)
      , Timeout(Timeout)
      , StartDelta(StartDelta)
      , StopDelta(StopDelta)
      , Instructions(std::move(Instructions)) {
    assert(!Metrics.empty());
  }

  ~CLIArgumentProblem() override = default;

  /// Evaluate the given individual by switching the current payload, doing the measurement and returning the results.
  /// \arg Individual The indivudal that should be measured.
  /// \returns The map from all metrics to their respective summaries for the measured individual.
  auto metrics(std::vector<unsigned> const& Individual) -> measurement::MetricSummaries override {
    // increment evaluation idx
    incrementFevals();

    const auto Groups = firestarter::InstructionGroups::fromInstructionAndValues(Instructions, Individual);

    // change the payload
    ChangePayloadFunction(Groups);

    // start the measurement
    // NOTE: starting the measurement must happen after switching to not
    // mess up ipc-estimate metric
    MeasurementWorker->startMeasurement();

    // wait for the measurement to finish
    std::this_thread::sleep_for(Timeout);

    // TODO(Issue #82): This is an ugly workaround for the ipc-estimate metric.
    // Changing the payload triggers a write of the iteration counter of
    // the last payload, which we use to estimate the ipc.
    ChangePayloadFunction(Groups);

    // return the results
    return MeasurementWorker->getValues(StartDelta, StopDelta);
  }

  /// Calculate the fitness based on the metric summaries of an individual. This will select the metrics that are
  /// required for the optimization, round them and potentially invert the results if the optimization metric name
  /// starts with a dash ('-').
  /// \arg Summaries The metric values for all metrics for an individual
  /// \return The vector containing the fitness for that metrics that are used in the optimization.
  [[nodiscard]] auto fitness(measurement::MetricSummaries const& Summaries) const -> std::vector<double> override {
    std::vector<double> Values = {};

    for (auto const& MetricName : OptimizationMetrics) {
      auto Summary = Summaries.at(MetricName);

      // round to two decimal places after the comma
      auto Value = std::round(Summary.Average * 100.0) / 100.0;

      // invert metric
      if (MetricName.inverted()) {
        Value *= -1.0;
      }

      Values.push_back(Value);
    }

    return Values;
  }

  /// Get the bounds of the problem. We currently set these bounds fix to a range from 0 to 100 for every instruction.
  /// \returns A vector the size of the number of instruction groups containing a tuple(0, 100).
  [[nodiscard]] auto getBounds() const -> std::vector<std::tuple<unsigned, unsigned>> override {
    std::vector<std::tuple<unsigned, unsigned>> Vec(Instructions.size(), std::make_tuple<unsigned, unsigned>(0, 100));

    return Vec;
  }

  /// Get the number of optimization objectives.
  [[nodiscard]] auto getNobjs() const -> std::size_t override { return OptimizationMetrics.size(); }
};

} // namespace firestarter::optimizer::problem
