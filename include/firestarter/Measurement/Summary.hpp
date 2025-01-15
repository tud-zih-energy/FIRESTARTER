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

#pragma once

#include "firestarter/Measurement/MetricInterface.h"
#include "firestarter/Measurement/TimeValue.hpp"

#include <chrono>
#include <nlohmann/json.hpp>
#include <vector>

namespace firestarter::measurement {

/// This struct summarized multiple timevalues. The duration, the number of time points an average and stddev is saved.
struct Summary {
  size_t NumTimepoints;
  std::chrono::milliseconds Duration;

  double Average;
  double Stddev;

  /// Calculate the summary over a range of timevalues for a given metric and number of threads.
  /// \arg Begin The start of the iterator
  /// \arg End The end of the iterator
  /// \arg MetricType This describes what each timevalue represents and how the metric needs to be calucated into a
  /// summary.
  /// \arg NumThreads The number of threads this metric was accumulated across.
  /// \returns The summary over the range of timevalues from a specific metric.
  static auto calculate(std::vector<TimeValue>::iterator Begin, std::vector<TimeValue>::iterator End,
                        MetricType MetricType, uint64_t NumThreads) -> Summary;
};

} // namespace firestarter::measurement
