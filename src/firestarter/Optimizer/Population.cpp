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

#include <firestarter/Logging/Log.hpp>
#include <firestarter/Optimizer/Population.hpp>

#include <algorithm>
#include <cassert>

using namespace firestarter::optimizer;

Population::Population(std::unique_ptr<Problem> &&problem,
                       std::size_t populationSize)
    : _problem(std::move(problem)), gen(rd()), random_distribution(1) {
  firestarter::log::trace() << "Generating " << populationSize
                            << " random individuals for initial population.";

  for (decltype(populationSize) i = 0; i < populationSize; i++) {
    this->append(this->getRandomIndividual());
  }
}

void Population::append(std::vector<unsigned> const &ind) {
  assert(this->problem().getDims() == ind.size());

  auto metrics = this->_problem->metrics(ind);
  auto fitness = this->_problem->fitness(metrics);

  this->append(ind, fitness);
}

void Population::append(std::vector<unsigned> const &ind,
                        std::vector<double> const &fit) {
  std::stringstream ss;
  ss << "  - Fitness: ";
  for (auto const &v : fit) {
    ss << v << " ";
  }
  firestarter::log::trace() << ss.str();

  assert(this->problem().getNobjs() == fit.size());
  assert(this->problem().getDims() == ind.size());

  auto id = this->getRandId();

  this->_individuals.push_back(std::make_tuple(id, ind, fit));
}

std::vector<unsigned> Population::getRandomIndividual() {
  auto dims = this->problem().getDims();
  auto const bounds = this->problem().getBounds();

  firestarter::log::trace() << "Generating random individual of size: " << dims;

  std::vector<unsigned> out(dims);

  for (decltype(dims) i = 0; i < dims; i++) {
    auto const lb = std::get<0>(bounds[i]);
    auto const ub = std::get<1>(bounds[i]);

    out[i] = std::uniform_int_distribution<unsigned>(lb, ub)(this->gen);

    firestarter::log::trace()
        << "  - " << i << ": [" << lb << "," << ub << "]: " << out[i];
  }

  return out;
}

std::vector<unsigned> const &Population::bestIndividual() const {
  // return an empty vector if the problem is multi objective, as there is no
  // single best individual
  if (this->problem().isMO()) {
    return std::vector<unsigned>();
  }

  // assert that we have individuals
  assert(this->_individuals.size() > 0);

  auto best = std::max_element(
      this->_individuals.begin(), this->_individuals.end(), [](auto a, auto b) {
        return std::get<2>(a).front() < std::get<2>(b).front();
      });

  assert(best != this->_individuals.end());

  return std::get<1>(*best);
}

// TODO: save our population into json perhaps for plotting nice graphs after
// the optimization
void Population::save() {
  assert(false);

  return;
}
