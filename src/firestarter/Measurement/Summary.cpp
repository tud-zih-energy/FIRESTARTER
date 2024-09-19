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
Summary Summary::calculate(std::vector<TimeValue>::iterator begin, std::vector<TimeValue>::iterator end,
                           metric_type_t metricType, unsigned long long numThreads) {
  std::vector<TimeValue> values = {};

  // TODO: i would really like to make this code a bit more readable, but i
  // could not find a way yet.
  if (metricType.accumalative) {
    TimeValue prev;

    if (begin != end) {
      prev = *begin++;
      for (auto it = begin; it != end; ++it) {
        auto time_diff =
            1e-6 * (double)std::chrono::duration_cast<std::chrono::microseconds>(it->time - prev.time).count();
        auto value_diff = it->value - prev.value;

        double value = value_diff / time_diff;

        if (metricType.divide_by_thread_count) {
          value /= numThreads;
        }

        values.push_back(TimeValue(prev.time, value));
        prev = *it;
      }
    }
  } else if (metricType.absolute) {
    for (auto it = begin; it != end; ++it) {
      double value = it->value;

      if (metricType.divide_by_thread_count) {
        value /= numThreads;
      }

      values.push_back(TimeValue(it->time, value));
    }
  } else {
    assert(false);
  }

  begin = values.begin();
  end = values.end();

  Summary summary{};

  summary.num_timepoints = std::distance(begin, end);

  if (summary.num_timepoints > 0) {

    auto last = begin;
    std::advance(last, summary.num_timepoints - 1);
    summary.duration = std::chrono::duration_cast<std::chrono::milliseconds>(last->time - begin->time);

    auto sum_over_nths = [&begin, end, summary](auto fn) {
      double acc = 0.0;
      for (auto it = begin; it != end; ++it) {
        acc += fn(it->value);
      }
      return acc / summary.num_timepoints;
    };

    summary.average = sum_over_nths([](double v) { return v; });
    summary.stddev = std::sqrt(sum_over_nths([&summary](double v) {
      double centered = v - summary.average;
      return centered * centered;
    }));
  }

  return summary;
}
