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

#include <firestarter/Measurement/Summary.hpp>

#include <cassert>
#include <cmath>

namespace firestarter::measurement {

// this functions borows a lot of code from
// https://github.com/metricq/metricq-cpp/blob/master/tools/metricq-summary/src/summary.cpp
auto Summary::calculate(std::vector<TimeValue>::iterator Begin, std::vector<TimeValue>::iterator End,
                        MetricType MetricType, uint64_t NumThreads) -> Summary {
  std::vector<TimeValue> Values = {};

  // TODO: i would really like to make this code a bit more readable, but i
  // could not find a way yet.
  if (MetricType.Accumalative) {
    TimeValue Prev;

    if (Begin != End) {
      Prev = *Begin++;
      for (auto It = Begin; It != End; ++It) {
        auto TimeDiff = 1e-6 * static_cast<double>(
                                   std::chrono::duration_cast<std::chrono::microseconds>(It->Time - Prev.Time).count());
        auto ValueDiff = It->Value - Prev.Value;

        double Value = ValueDiff / TimeDiff;

        if (MetricType.DivideByThreadCount) {
          Value /= NumThreads;
        }

        Values.emplace_back(Prev.Time, Value);
        Prev = *It;
      }
    }
  } else if (MetricType.Absolute) {
    for (auto It = Begin; It != End; ++It) {
      double Value = It->Value;

      if (MetricType.DivideByThreadCount) {
        Value /= NumThreads;
      }

      Values.emplace_back(It->Time, Value);
    }
  } else {
    assert(false);
  }

  Begin = Values.begin();
  End = Values.end();

  Summary SummaryVal{};

  SummaryVal.NumTimepoints = std::distance(Begin, End);

  if (SummaryVal.NumTimepoints > 0) {

    auto Last = Begin;
    std::advance(Last, SummaryVal.NumTimepoints - 1);
    SummaryVal.Duration = std::chrono::duration_cast<std::chrono::milliseconds>(Last->Time - Begin->Time);

    auto SumOverNths = [&Begin, End, SummaryVal](auto Fn) {
      double Acc = 0.0;
      for (auto It = Begin; It != End; ++It) {
        Acc += Fn(It->Value);
      }
      return Acc / SummaryVal.NumTimepoints;
    };

    SummaryVal.Average = SumOverNths([](double V) { return V; });
    SummaryVal.Stddev = std::sqrt(SumOverNths([&SummaryVal](double V) {
      double Centered = V - SummaryVal.Average;
      return Centered * Centered;
    }));
  }

  return SummaryVal;
}

} // namespace firestarter::measurement