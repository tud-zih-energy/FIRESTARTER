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

#include "firestarter/Optimizer/Population.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/Optimizer/History.hpp"

#include <cassert>
#include <random>

namespace firestarter::optimizer {

void Population::generateInitialPopulation(std::size_t PopulationSize) {
  firestarter::log::trace() << "Generating " << PopulationSize << " random individuals for initial population.";

  auto Dims = this->problem().getDims();
  auto Remaining = PopulationSize;

  if (!(PopulationSize < Dims)) {
    for (decltype(Dims) I = 0; I < Dims; I++) {
      Individual Vec(Dims, 0);
      Vec[I] = 1;
      this->append(Vec);
      Remaining--;
    }
  } else {
    firestarter::log::trace() << "Population size (" << std::to_string(PopulationSize)
                              << ") is less than size of problem dimension (" << std::to_string(Dims) << ")";
  }

  for (decltype(Remaining) I = 0; I < Remaining; I++) {
    this->append(this->getRandomIndividual());
  }
}

auto Population::size() const -> std::size_t { return X.size(); }

void Population::append(Individual const& Ind) {
  assert(this->problem().getDims() == Ind.size());

  std::map<std::string, firestarter::measurement::Summary> Metrics;

  // check if we already evaluated this individual
  const auto OptionalMetric = History::find(Ind);
  if (OptionalMetric.has_value()) {
    Metrics = OptionalMetric.value();
  } else {
    Metrics = this->ProblemPtr->metrics(Ind);
  }

  auto Fitness = this->ProblemPtr->fitness(Metrics);

  this->append(Ind, Fitness);

  if (!OptionalMetric.has_value()) {
    History::append(Ind, Metrics);
  }
}

void Population::append(Individual const& Ind, std::vector<double> const& Fit) {
  std::stringstream Ss;
  Ss << "  - Fitness: ";
  for (auto const& V : Fit) {
    Ss << V << " ";
  }
  firestarter::log::trace() << Ss.str();

  assert(this->problem().getNobjs() == Fit.size());
  assert(this->problem().getDims() == Ind.size());

  this->X.push_back(Ind);
  this->F.push_back(Fit);
}

void Population::insert(std::size_t Idx, Individual const& Ind, std::vector<double> const& Fit) {
  // assert that population is big enough
  assert(X.size() > Idx);

  X[Idx] = Ind;
  F[Idx] = Fit;
}

auto Population::getRandomIndividual() const -> Individual {
  auto Dims = this->problem().getDims();
  auto const Bounds = this->problem().getBounds();

  std::random_device Rd;
  std::mt19937 Rng(Rd());

  firestarter::log::trace() << "Generating random individual of size: " << Dims;

  Individual Out(Dims);

  for (decltype(Dims) I = 0; I < Dims; I++) {
    auto const Lb = std::get<0>(Bounds[I]);
    auto const Ub = std::get<1>(Bounds[I]);

    Out[I] = std::uniform_int_distribution<unsigned>(Lb, Ub)(Rng);

    firestarter::log::trace() << "  - " << I << ": [" << Lb << "," << Ub << "]: " << Out[I];
  }

  return Out;
}

} // namespace firestarter::optimizer