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

#include "../Measurement/Summary.hpp"
#include "Individual.hpp"
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

namespace firestarter::optimizer {

/// This class models the abstract problem which should be optimized. It provides the methods to evaluate an individual
/// and calculate its fitness.
class Problem {
  /// The number of metric evaluations
  uint64_t Fevals = 0;

public:
  Problem() = default;
  virtual ~Problem() = default;

  /// Perform an evaluation of the supplied individual. This returns a map from the metric name to their respective
  /// summary. This function will increment the fevals.
  /// \arg Individual The individual that should be evaluated.
  /// \returns A map from metric name to the summary of this metric for the specific individual
  virtual auto metrics(Individual const& Individual) -> std::map<std::string, firestarter::measurement::Summary> = 0;

  /// Convert the result of one evaluation into a fitness (vector of doubles) for the supplied summaries
  /// \arg Summaries The summaries of one evaluation.
  /// \returns The fitness vector derived from the summaries. The size of this vector is equal to the number of
  /// objectives.
  [[nodiscard]] virtual auto fitness(std::map<std::string, firestarter::measurement::Summary> const& Summaries) const
      -> std::vector<double> = 0;

  /// Get the bounds of the problem. For each dimension a min and max value is supplied.
  /// \return The min and max bound per dimension.
  [[nodiscard]] virtual auto getBounds() const -> std::vector<std::tuple<unsigned, unsigned>> = 0;

  /// Get the number of dimensions of the problem.
  /// \returns The number of dimensions.
  [[nodiscard]] auto getDims() const -> std::size_t { return this->getBounds().size(); };

  /// Get the number of optimization objectives for this problem.
  /// \arg The number of objectives.
  [[nodiscard]] virtual auto getNobjs() const -> std::size_t = 0;

  /// Check if the problem is a multi-objective one.
  [[nodiscard]] auto isMO() const -> bool { return this->getNobjs() > 1; };

  /// Get the number of evaluations.
  [[nodiscard]] auto getFevals() const -> uint64_t { return Fevals; };

protected:
  /// Increment the number of evaluations.
  void incrementFevals() { Fevals++; };
};

} // namespace firestarter::optimizer
