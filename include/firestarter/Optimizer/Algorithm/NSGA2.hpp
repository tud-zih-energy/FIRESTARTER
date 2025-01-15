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

#include "firestarter/Optimizer/Algorithm.hpp"

namespace firestarter::optimizer::algorithm {

/// This class implements the NSGA2 evolutionary optimization algorithm.
/// The NSGA2 algorithm, as described in "A fast and elitist multiobjective genetic algorithm: NSGA-II"
/// (https://dl.acm.org/doi/10.1109/4235.996017), is a multiobjective algorithm allowing FIRESTARTER to optimize with
/// two (or more) metrics. This is relevant because adding the IPC (instruction per cycle) metric supports the
/// optimization algorithm to converge towards higher power consumption.
class NSGA2 : public Algorithm {
public:
  /// Initialize the NSGA2 algorithm.
  /// \arg Gen The number of generation that the algorithm uses to evolve its population.
  /// \arg Cr The Crossover probability. Must be in range [0,1[
  /// \arg M Mutation probability. Must be in range [0,1]
  NSGA2(unsigned Gen, double Cr, double M);
  ~NSGA2() override = default;

  /// Check if the problem and population size matches the requirements of NSGA2. We must have a multi-objective problem
  /// and at least 5 and a multiple of 4 individuals in our population.
  /// \arg Prob The poblem that should be optimized with this algorithm
  /// \arg PopulationSize The initial size of the population that is used
  void check(firestarter::optimizer::Problem const& Prob, std::size_t PopulationSize) override;

  /// Evolve the population across multiple iterations.
  /// \arg Pop The initial population
  /// \returns The final population after the optimization has run
  auto evolve(firestarter::optimizer::Population& Pop) -> firestarter::optimizer::Population override;

private:
  // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)

  /// The number of generations of the NSGA2 algorithm.
  const unsigned Gen;
  /// The crossover propability in the range [0,1[.
  const double Cr;
  /// The mutation propability in the range [0,1].
  const double M;

  // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace firestarter::optimizer::algorithm
