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

#include <firestarter/Optimizer/Problem.hpp>

#include <cstring>
#include <random>
#include <tuple>
#include <vector>

namespace firestarter::optimizer {

class Population {
public:
  // Construct a population from a problem.
  // population size is given by parameter.
  Population(Problem &&problem, std::size_t populationSize = 0);

  ~Population();

  // add one individual to the population. fitness will be evaluated.
  void append(std::vector<unsigned> const &ind);
  // add one individual to the population with a fitness.
  void append(std::vector<unsigned> const &ind, std::vector<double> const &fit);

  // get a random individual inside bounds of problem
  std::vector<unsigned> getRandomIndividual() const;

  // return the best individual in case of single-objective.
  // returns nothing, if the problem is multi-objective
  std::vector<unsigned> const &bestIndividual() const;

  // save the population
  void save();

  Problem &problem() { return _problem; }

  std::vector<std::tuple<unsigned long long, std::vector<unsigned>,
                         std::vector<double>>> const &
  individuals() const {
    return _individuals;
  }

private:
  // our problem.
  Problem _problem;
  // a vector containing a tuple of id, individual and its fitness
  std::vector<std::tuple<unsigned long long, std::vector<unsigned>,
                         std::vector<double>>>
      _individuals;

  unsigned long long getRandId() {
    return this->random_distribution(this->gen);
  }

  std::random_device rd;
  std::mt19937 gen;
  std::uniform_int_distribution<unsigned long long> random_distribution;
};

} // namespace firestarter::optimizer

#endif
