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

#include "Individual.hpp"
#include "Problem.hpp"
#include <cstring>
#include <memory>
#include <vector>

namespace firestarter::optimizer {

/// This class models the notion of a population used by the NSGA2 algorithm that contains a number of individuals with
/// their associated fitness.
class Population {
public:
  Population() = delete;

  /// Construct a population from a problem.
  explicit Population(std::shared_ptr<Problem>&& ProblemPtr)
      : ProblemPtr(std::move(ProblemPtr)) {}

  ~Population() = default;

  /// Generate a supplied number of individuals and save them with their fitness in this datastructure. If the number is
  /// less then the number of dimensions we fill them with random individuals. If it is at least the number of
  /// dimension, we first create individuals with one dimension equal to one and the rest equal to zero.
  /// \arg PopulationSize The number of individuals to generate.
  void generateInitialPopulation(std::size_t PopulationSize);

  /// The number of individuals in this population.
  [[nodiscard]] auto size() const -> std::size_t;

  /// Append one individual to the population. If a lookup of the fitness in the history is no successful, the
  /// individual will be evaluated and the fitness saved.
  /// \arg Ind The individual to be added to the population.
  void append(Individual const& Ind);

  /// Insert an indiviudal and an associated fitness at a specific index in the population.
  /// \arg Idx On which index to insert in the population.
  /// \arg Ind The individual to insert.
  /// \arg Fit The fitness to insert.
  void insert(std::size_t Idx, Individual const& Ind, std::vector<double> const& Fit);

  /// Generate a random individual inside the bounds of the problem based on a non-determenistic generator.
  /// \returns The random individual inside the bounds of the problem.
  [[nodiscard]] auto getRandomIndividual() const -> Individual;

  /// Const reference to the optimization problem.
  [[nodiscard]] auto problem() const -> Problem const& { return *ProblemPtr; }

  /// Const reference to the vector of individuals.
  [[nodiscard]] auto x() const -> std::vector<Individual> const& { return X; }
  /// Const reference to the vector of fitnesses.
  [[nodiscard]] auto f() const -> std::vector<std::vector<double>> const& { return F; }

private:
  /// Append one individual with a given fitness to the population.
  /// \arg Ind The individual to be appended to the population.
  /// \arg Fit The fitness of the individual.
  void append(Individual const& Ind, std::vector<double> const& Fit);

  /// The optimization problem
  std::shared_ptr<Problem> ProblemPtr;

  /// The vector of individuals
  std::vector<Individual> X;
  /// The vector of fitnesses associated to each individual
  std::vector<std::vector<double>> F;
};

} // namespace firestarter::optimizer