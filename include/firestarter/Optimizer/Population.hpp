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

#ifndef FIRESTARTER_OPTIMIZER_POPULATION_HPP
#define FIRESTARTER_OPTIMIZER_POPULATION_HPP

#include <cstring>
#include <firestarter/Optimizer/History.hpp>
#include <firestarter/Optimizer/Individual.hpp>
#include <firestarter/Optimizer/Problem.hpp>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace firestarter::optimizer {

class Population {
public:
  // Construct a population from a problem.
  Population() = default;

  explicit Population(std::shared_ptr<Problem>&& ProblemPtr)
      : ProblemPtr(std::move(ProblemPtr))
      , Gen(Rd()) {}

  Population(Population& Pop)
      : ProblemPtr(Pop.ProblemPtr)
      , X(Pop.X)
      , F(Pop.F)
      , Gen(Rd()) {}

  auto operator=(Population const& Pop) -> Population& {
    ProblemPtr = Pop.ProblemPtr;
    X = Pop.X;
    F = Pop.F;
    Gen = Pop.Gen;

    return *this;
  }

  ~Population() = default;

  void generateInitialPopulation(std::size_t PopulationSize = 0);

  [[nodiscard]] auto size() const -> std::size_t;

  // add one individual to the population. fitness will be evaluated.
  void append(Individual const& Ind);

  void insert(std::size_t Idx, Individual const& Ind, std::vector<double> const& Fit);

  // get a random individual inside bounds of problem
  auto getRandomIndividual() -> Individual;

  // returns the best individual in case of single-objective.
  // return nothing in case of mutli-objective.
  [[nodiscard]] auto bestIndividual() const -> std::optional<Individual>;

  [[nodiscard]] auto problem() const -> Problem const& { return *ProblemPtr; }

  [[nodiscard]] auto x() const -> std::vector<Individual> const& { return X; }
  [[nodiscard]] auto f() const -> std::vector<std::vector<double>> const& { return F; }

private:
  // add one individual to the population with a fitness.
  void append(Individual const& Ind, std::vector<double> const& Fit);

  // our problem.
  std::shared_ptr<Problem> ProblemPtr;

  std::vector<Individual> X;
  std::vector<std::vector<double>> F;

  std::random_device Rd;
  std::mt19937 Gen;
};

} // namespace firestarter::optimizer

#endif
