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

#include "../../Measurement/MeasurementWorker.hpp"
#include "../Problem.hpp"
#include <cassert>
#include <functional>
#include <thread>
#include <tuple>
#include <utility>

namespace firestarter::optimizer::problem {

class CLIArgumentProblem final : public firestarter::optimizer::Problem {
public:
  CLIArgumentProblem(std::function<void(std::vector<std::pair<std::string, unsigned>> const&)>&& ChangePayloadFunction,
                     std::shared_ptr<firestarter::measurement::MeasurementWorker> const& MeasurementWorker,
                     std::vector<std::string> const& Metrics, std::chrono::seconds Timeout,
                     std::chrono::milliseconds StartDelta, std::chrono::milliseconds StopDelta,
                     std::vector<std::string> const& InstructionGroups)
      : ChangePayloadFunction(ChangePayloadFunction)
      , MeasurementWorker(MeasurementWorker)
      , Metrics(Metrics)
      , Timeout(Timeout)
      , StartDelta(StartDelta)
      , StopDelta(StopDelta)
      , InstructionGroups(InstructionGroups) {
    assert(Metrics.size() != 0);
  }

  ~CLIArgumentProblem() override = default;

  // return all available metrics for the individual
  auto metrics(std::vector<unsigned> const& Individual)
      -> std::map<std::string, firestarter::measurement::Summary> override {
    // increment evaluation idx
    Fevals++;

    // change the payload
    assert(InstructionGroups.size() == Individual.size());
    std::vector<std::pair<std::string, unsigned>> Payload = {};
    auto It1 = InstructionGroups.begin();
    auto It2 = Individual.begin();
    for (; It1 != InstructionGroups.end(); ++It1, ++It2) {
      Payload.emplace_back(*It1, *It2);
    }
    ChangePayloadFunction(Payload);

    // start the measurement
    // NOTE: starting the measurement must happen after switching to not
    // mess up ipc-estimate metric
    MeasurementWorker->startMeasurement();

    // wait for the measurement to finish
    std::this_thread::sleep_for(Timeout);

    // FIXME: this is an ugly workaround for the ipc-estimate metric
    // changeing the payload triggers a write of the iteration counter of
    // the last payload, which we use to estimate the ipc.
    ChangePayloadFunction(Payload);

    // return the results
    return MeasurementWorker->getValues(StartDelta, StopDelta);
  }

  auto fitness(std::map<std::string, firestarter::measurement::Summary> const& Summaries)
      -> std::vector<double> override {
    std::vector<double> Values = {};

    for (auto const& MetricName : Metrics) {
      auto FindName = [MetricName](auto const& Summary) {
        auto InvertedName = "-" + Summary.first;
        return MetricName.compare(Summary.first) == 0 || MetricName.compare(InvertedName) == 0;
      };

      auto It = std::find_if(Summaries.begin(), Summaries.end(), FindName);

      if (It == Summaries.end()) {
        continue;
      }

      // round to two decimal places after the comma
      auto Value = std::round(It->second.Average * 100.0) / 100.0;

      // invert metric
      if (MetricName[0] == '-') {
        Value *= -1.0;
      }

      Values.push_back(Value);
    }

    return Values;
  }

  // get the bounds of the problem
  [[nodiscard]] auto getBounds() const -> std::vector<std::tuple<unsigned, unsigned>> override {
    std::vector<std::tuple<unsigned, unsigned>> Vec(InstructionGroups.size(),
                                                    std::make_tuple<unsigned, unsigned>(0, 100));

    return Vec;
  }

  // get the number of objectives.
  [[nodiscard]] auto getNobjs() const -> std::size_t override { return Metrics.size(); }

private:
  std::function<void(std::vector<std::pair<std::string, unsigned>> const&)> ChangePayloadFunction;
  std::shared_ptr<firestarter::measurement::MeasurementWorker> MeasurementWorker;
  std::vector<std::string> Metrics;
  std::chrono::seconds Timeout;
  std::chrono::milliseconds StartDelta;
  std::chrono::milliseconds StopDelta;
  std::vector<std::string> InstructionGroups;
};

} // namespace firestarter::optimizer::problem
