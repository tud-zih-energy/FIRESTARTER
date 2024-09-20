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

#include <firestarter/Logging/Log.hpp>
#include <firestarter/Optimizer/Population.hpp>

#include <algorithm>
#include <cassert>
#include <stdexcept>

using namespace firestarter::optimizer;

void Population::generateInitialPopulation(std::size_t populationSize) {
  firestarter::log::trace() << "Generating " << populationSize
                            << " random individuals for initial population.";

  auto dims = this->problem().getDims();
  auto remaining = populationSize;

  if (!(populationSize < dims)) {
    for (decltype(dims) i = 0; i < dims; i++) {
      Individual vec(dims, 0);
      vec[i] = 1;
      this->append(vec);
    }

    remaining -= dims;
  } else {
    firestarter::log::trace()
        << "Population size (" << std::to_string(populationSize)
        << ") is less than size of problem dimension (" << std::to_string(dims)
        << ")";
  }

  for (decltype(remaining) i = 0; i < remaining; i++) {
    this->append(this->getRandomIndividual());
  }
}

std::size_t Population::size() const { return _x.size(); }

void Population::append(Individual const &ind) {
  assert(this->problem().getDims() == ind.size());

  std::map<std::string, firestarter::measurement::Summary> metrics;

  // check if we already evaluated this individual
  if (_saveHistory) {
    auto optional_metric = History::find(ind);
    if (optional_metric.has_value()) {
      metrics = optional_metric.value();
    } else {
      metrics = this->_problem->metrics(ind);
    }

    auto fitness = this->_problem->fitness(metrics);

    this->append(ind, fitness);

    if (!optional_metric.has_value()) {
      History::append(ind, metrics);
    }
  } else {
    this->append(ind, this->_problem->fitness(this->_problem->metrics(ind)));
  }
}

void Population::append(Individual const &ind, std::vector<double> const &fit) {
  std::stringstream ss;
  ss << "  - Fitness: ";
  for (auto const &v : fit) {
    ss << v << " ";
  }
  // firestarter::log::trace() << ss.str();

  assert(this->problem().getNobjs() == fit.size());
  assert(this->problem().getDims() == ind.size());

  this->_x.push_back(ind);
  this->_f.push_back(fit);
}

void Population::insert(std::size_t idx, Individual const &ind,
                        std::vector<double> const &fit) {
  // assert that population is big enough
  assert(_x.size() > idx);

  _x[idx] = ind;
  _f[idx] = fit;
}

Individual Population::getRandomIndividual() {
  auto dims = this->problem().getDims();
  auto const bounds = this->problem().getBounds();

  // firestarter::log::trace() << "Generating random individual of size: " <<
  // dims;

  Individual out(dims);

  for (decltype(dims) i = 0; i < dims; i++) {
    auto const lb = std::get<0>(bounds[i]);
    auto const ub = std::get<1>(bounds[i]);

    out[i] = std::uniform_int_distribution<unsigned>(lb, ub)(this->gen);

    // firestarter::log::trace()
    //    << "  - " << i << ": [" << lb << "," << ub << "]: " << out[i];
  }

  return out;
}

std::optional<Individual> Population::bestIndividual() const {
  // return an empty vector if the problem is multi objective, as there is no
  // single best individual
  if (this->problem().isMO()) {
    return {};
  }

  // assert that we have individuals
  assert(this->_x.size() > 0);

  auto best = std::max_element(this->_x.begin(), this->_x.end(),
                               [](auto a, auto b) { return a < b; });

  assert(best != this->_x.end());

  return *best;
}
