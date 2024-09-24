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

// This file borrows a lot of code from https://github.com/esa/pagmo2

#include <firestarter/Optimizer/Algorithm/NSGA2.hpp>
#include <firestarter/Optimizer/Individual.hpp>
#include <firestarter/Optimizer/Util/MultiObjective.hpp>

#include <algorithm>
#include <stdexcept>

using namespace firestarter::optimizer::algorithm;

NSGA2::NSGA2(unsigned Gen, double Cr, double M)
    : Gen(Gen)
    , Cr(Cr)
    , M(M) {
  if (Cr >= 1. || Cr < 0.) {
    throw std::invalid_argument("The crossover probability must be in the "
                                "[0,1[ range, while a value of " +
                                std::to_string(Cr) + " was detected");
  }
  if (M < 0. || M > 1.) {
    throw std::invalid_argument("The mutation probability must be in the [0,1] "
                                "range, while a value of " +
                                std::to_string(M) + " was detected");
  }
}

void NSGA2::checkPopulation(firestarter::optimizer::Population const& Pop, std::size_t PopulationSize) {
  const auto& Prob = Pop.problem();

  if (!Prob.isMO()) {
    throw std::invalid_argument("NSGA2 is a multiobjective algorithms, while number of objectives is " +
                                std::to_string(Prob.getNobjs()));
  }

  if (PopulationSize < 5u || (PopulationSize % 4 != 0u)) {
    throw std::invalid_argument("for NSGA-II at least 5 individuals in the "
                                "population are needed and the "
                                "population size must be a multiple of 4. "
                                "Detected input population size is: " +
                                std::to_string(PopulationSize));
  }
}

auto NSGA2::evolve(firestarter::optimizer::Population& Pop) -> firestarter::optimizer::Population {
  const auto& prob = Pop.problem();
  const auto bounds = prob.getBounds();
  auto NP = Pop.size();
  auto fevals0 = prob.getFevals();

  this->checkPopulation(const_cast<firestarter::optimizer::Population const&>(Pop), NP);

  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<Individual::size_type> best_idx(NP), shuffle1(NP), shuffle2(NP);
  Individual::size_type parent1_idx, parent2_idx;
  std::pair<Individual, Individual> children;

  std::iota(shuffle1.begin(), shuffle1.end(), Individual::size_type(0));
  std::iota(shuffle2.begin(), shuffle2.end(), Individual::size_type(0));

  {
    std::stringstream ss;

    ss << std::endl << std::setw(7) << "Gen:" << std::setw(15) << "Fevals:";
    for (decltype(prob.getNobjs()) i = 0; i < prob.getNobjs(); ++i) {
      ss << std::setw(15) << "ideal" << std::to_string(i + 1u) << ":";
    }
    firestarter::log::info() << ss.str();
  }

  for (decltype(Gen) gen = 1u; gen <= Gen; ++gen) {
    {
      // Print the logs
      std::vector<double> idealPoint = util::ideal(Pop.f());
      std::stringstream ss;

      ss << std::setw(7) << gen << std::setw(15) << prob.getFevals() - fevals0;
      for (decltype(idealPoint.size()) i = 0; i < idealPoint.size(); ++i) {
        ss << std::setw(15) << idealPoint[i];
      }

      firestarter::log::info() << ss.str();
    }

    // At each generation we make a copy of the population into popnew
    firestarter::optimizer::Population popnew(Pop);

    // We create some pseudo-random permutation of the poulation indexes
    std::random_shuffle(shuffle1.begin(), shuffle1.end());
    std::random_shuffle(shuffle2.begin(), shuffle2.end());

    // We compute crowding distance and non dominated rank for the current
    // population
    auto fnds_res = util::fastNonDominatedSorting(Pop.f());
    auto ndf = std::get<0>(fnds_res); // non dominated fronts [[0,3,2],[1,5,6],[4],...]
    std::vector<double> pop_cd(NP);   // crowding distances of the whole population
    auto ndr = std::get<3>(fnds_res); // non domination rank [0,1,0,0,2,1,1, ... ]
    for (const auto& front_idxs : ndf) {
      if (front_idxs.size() == 1u) { // handles the case where the front has collapsed to one point
        pop_cd[front_idxs[0]] = std::numeric_limits<double>::infinity();
      } else if (front_idxs.size() == 2u) { // handles the case where the front
        // has collapsed to one point
        pop_cd[front_idxs[0]] = std::numeric_limits<double>::infinity();
        pop_cd[front_idxs[1]] = std::numeric_limits<double>::infinity();
      } else {
        std::vector<std::vector<double>> front;
        for (auto idx : front_idxs) {
          front.push_back(Pop.f()[idx]);
        }
        auto cd = util::crowdingDistance(front);
        for (decltype(cd.size()) i = 0u; i < cd.size(); ++i) {
          pop_cd[front_idxs[i]] = cd[i];
        }
      }
    }

    // We then loop thorugh all individuals with increment 4 to select two pairs
    // of parents that will each create 2 new offspring
    for (decltype(NP) i = 0u; i < NP; i += 4) {
      // We create two offsprings using the shuffled list 1
      parent1_idx = util::moTournamentSelection(shuffle1[i], shuffle1[i + 1], ndr, pop_cd, rng);
      parent2_idx = util::moTournamentSelection(shuffle1[i + 2], shuffle1[i + 3], ndr, pop_cd, rng);
      children = util::sbxCrossover(Pop.x()[parent1_idx], Pop.x()[parent2_idx], Cr, rng);
      util::polynomialMutation(children.first, bounds, M, rng);
      util::polynomialMutation(children.second, bounds, M, rng);

      popnew.append(children.first);
      popnew.append(children.second);

      // We repeat with the shuffled list 2
      parent1_idx = util::moTournamentSelection(shuffle2[i], shuffle2[i + 1], ndr, pop_cd, rng);
      parent2_idx = util::moTournamentSelection(shuffle2[i + 2], shuffle2[i + 3], ndr, pop_cd, rng);
      children = util::sbxCrossover(Pop.x()[parent1_idx], Pop.x()[parent2_idx], Cr, rng);
      util::polynomialMutation(children.first, bounds, M, rng);
      util::polynomialMutation(children.second, bounds, M, rng);

      popnew.append(children.first);
      popnew.append(children.second);
    } // popnew now contains 2NP individuals
    // This method returns the sorted N best individuals in the population
    // according to the crowded comparison operator
    best_idx = util::selectBestNMo(popnew.f(), NP);
    // We insert into the population
    for (decltype(NP) i = 0; i < NP; ++i) {
      Pop.insert(i, popnew.x()[best_idx[i]], popnew.f()[best_idx[i]]);
    }
  }

  return Pop;
}
