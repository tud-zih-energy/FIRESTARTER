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

#include "firestarter/Optimizer/Population.hpp"

namespace firestarter::optimizer {

/// Abstract class to provide an interface for evolutionary optimization algorithms.
class Algorithm {
public:
  Algorithm() = default;
  virtual ~Algorithm() = default;

  /// Check if the population size and the problem matches the requirements of the algorithm. Asserts if this checks
  /// fail.
  /// \arg Prob The poblem that should be optimized with this algorithm
  /// \arg PopulationSize The initial size of the population that is used
  virtual void check(Problem const& Prob, std::size_t PopulationSize) = 0;

  /// Evolve the population across multiple iterations.
  /// \arg Pop The initial population
  /// \returns The final population after the optimization has run
  virtual auto evolve(Population& Pop) -> Population = 0;
};

} // namespace firestarter::optimizer
