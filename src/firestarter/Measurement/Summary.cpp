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

using namespace firestarter::measurement;

// this functions borows a lot of code from
// https://github.com/metricq/metricq-cpp/blob/master/tools/metricq-summary/src/summary.cpp
auto Summary::calculate(std::vector<TimeValue>::iterator Begin, std::vector<TimeValue>::iterator End,
                        MetricType MetricType, uint64_t NumThreads) -> Summary {
  std::vector<TimeValue> Values = {};

  // TODO: i would really like to make this code a bit more readable, but i
  // could not find a way yet.
  if (MetricType.Accumalative) {
    TimeValue prev;

    if (Begin != End) {
      prev = *Begin++;
      for (auto it = Begin; it != End; ++it) {
        auto time_diff =
            1e-6 * (double)std::chrono::duration_cast<std::chrono::microseconds>(it->Time - prev.Time).count();
        auto value_diff = it->Value - prev.Value;

        double value = value_diff / time_diff;

        if (MetricType.DivideByThreadCount) {
          value /= NumThreads;
        }

        Values.emplace_back(prev.Time, value);
        prev = *it;
      }
    }
  } else if (MetricType.Absolute) {
    for (auto it = Begin; it != End; ++it) {
      double value = it->Value;

      if (MetricType.DivideByThreadCount) {
        value /= NumThreads;
      }

      Values.emplace_back(it->Time, value);
    }
  } else {
    assert(false);
  }

  Begin = Values.begin();
  End = Values.end();

  Summary SummaryVal{};

  SummaryVal.NumTimepoints = std::distance(Begin, End);

  if (SummaryVal.NumTimepoints > 0) {

    auto last = Begin;
    std::advance(last, SummaryVal.NumTimepoints - 1);
    SummaryVal.Duration = std::chrono::duration_cast<std::chrono::milliseconds>(last->Time - Begin->Time);

    auto sum_over_nths = [&Begin, End, SummaryVal](auto fn) {
      double acc = 0.0;
      for (auto it = Begin; it != End; ++it) {
        acc += fn(it->Value);
      }
      return acc / SummaryVal.NumTimepoints;
    };

    SummaryVal.Average = sum_over_nths([](double v) { return v; });
    SummaryVal.Stddev = std::sqrt(sum_over_nths([&SummaryVal](double v) {
      double centered = v - SummaryVal.Average;
      return centered * centered;
    }));
  }

  return SummaryVal;
}
