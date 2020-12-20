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

#ifndef FIRESTARTER_OPTIMIZER_POPULATION_HPP
#define FIRESTARTER_OPTIMIZER_POPULATION_HPP

#include <firestarter/Optimizer/History.hpp>
#include <firestarter/Optimizer/Individual.hpp>
#include <firestarter/Optimizer/Problem.hpp>

#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

namespace firestarter::optimizer {

class Population {
public:
  // Construct a population from a problem.
  // population size is given by parameter.
  Population(std::shared_ptr<Problem> &&problem,
             std::size_t populationSize = 0);

  Population(Population &pop)
      : _problem(pop._problem), _x(pop._x), _f(pop._f), gen(rd()) {}

  ~Population() {}

  std::size_t size();

  // add one individual to the population. fitness will be evaluated.
  void append(Individual const &ind);

  void insert(std::size_t idx, Individual const &ind,
              std::vector<double> const &fit);

  // get a random individual inside bounds of problem
  Individual getRandomIndividual();

  // returns the best individual in case of single-objective.
  // return nothing in case of mutli-objective.
  std::optional<Individual> bestIndividual() const;

  Problem const &problem() const { return *_problem; }

  std::vector<Individual> const &x() const { return _x; }
  std::vector<std::vector<double>> const &f() const { return _f; }

private:
  // add one individual to the population with a fitness.
  void append(Individual const &ind, std::vector<double> const &fit);

  // our problem.
  std::shared_ptr<Problem> _problem;

  std::vector<Individual> _x;
  std::vector<std::vector<double>> _f;

  std::random_device rd;
  std::mt19937 gen;
};

} // namespace firestarter::optimizer

#endif
