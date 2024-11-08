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

#include <firestarter/Logging/Log.hpp>
#include <firestarter/Optimizer/Algorithm/NSGA2.hpp>
#include <firestarter/Optimizer/Individual.hpp>
#include <firestarter/Optimizer/Util/MultiObjective.hpp>

#include <algorithm>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace firestarter::optimizer::algorithm {

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

void NSGA2::check(firestarter::optimizer::Problem const& Prob, std::size_t PopulationSize) {
  if (!Prob.isMO()) {
    throw std::invalid_argument("NSGA2 is a multiobjective algorithms, while number of objectives is " +
                                std::to_string(Prob.getNobjs()));
  }

  if (PopulationSize < 5U || (PopulationSize % 4 != 0U)) {
    throw std::invalid_argument("for NSGA-II at least 5 individuals in the "
                                "population are needed and the "
                                "population size must be a multiple of 4. "
                                "Detected input population size is: " +
                                std::to_string(PopulationSize));
  }
}

auto NSGA2::evolve(firestarter::optimizer::Population& Pop) -> firestarter::optimizer::Population {
  const auto& Prob = Pop.problem();
  const auto Bounds = Prob.getBounds();
  auto NP = Pop.size();
  auto Fevals0 = Prob.getFevals();

  this->check(Prob, NP);

  std::random_device Rd;
  std::mt19937 Rng(Rd());

  std::vector<Individual::size_type> BestIdx(NP);
  std::vector<Individual::size_type> Shuffle1(NP);
  std::vector<Individual::size_type> Shuffle2(NP);
  Individual::size_type Parent1Idx = 0;
  Individual::size_type Parent2Idx = 0;
  std::pair<Individual, Individual> Children;

  std::iota(Shuffle1.begin(), Shuffle1.end(), static_cast<Individual::size_type>(0));
  std::iota(Shuffle2.begin(), Shuffle2.end(), static_cast<Individual::size_type>(0));

  {
    std::stringstream Ss;

    Ss << '\n' << std::setw(7) << "Gen:" << std::setw(15) << "Fevals:";
    for (decltype(Prob.getNobjs()) I = 0; I < Prob.getNobjs(); ++I) {
      Ss << std::setw(15) << "ideal" << std::to_string(I + 1U) << ":";
    }
    firestarter::log::info() << Ss.str();
  }

  for (auto I = 1U; I <= Gen; ++I) {
    {
      // Print the logs
      const auto IdealPoint = util::ideal(Pop.f());
      std::stringstream Ss;

      Ss << std::setw(7) << I << std::setw(15) << Prob.getFevals() - Fevals0;
      for (const auto I : IdealPoint) {
        Ss << std::setw(15) << I;
      }

      firestarter::log::info() << Ss.str();
    }

    // At each generation we make a copy of the population into popnew
    firestarter::optimizer::Population Popnew(Pop);

    // We create some pseudo-random permutation of the poulation indexes
    std::shuffle(Shuffle1.begin(), Shuffle1.end(), Rng);
    std::shuffle(Shuffle2.begin(), Shuffle2.end(), Rng);

    // We compute crowding distance and non dominated rank for the current
    // population
    auto FndsRes = util::fastNonDominatedSorting(Pop.f());
    auto Ndf = std::get<0>(FndsRes); // non dominated fronts [[0,3,2],[1,5,6],[4],...]
    std::vector<double> PopCd(NP);   // crowding distances of the whole population
    auto Ndr = std::get<3>(FndsRes); // non domination rank [0,1,0,0,2,1,1, ... ]
    for (const auto& FrontIdxs : Ndf) {
      if (FrontIdxs.size() == 1U) { // handles the case where the front has collapsed to one point
        PopCd[FrontIdxs[0]] = std::numeric_limits<double>::infinity();
      } else if (FrontIdxs.size() == 2U) { // handles the case where the front
        // has collapsed to one point
        PopCd[FrontIdxs[0]] = std::numeric_limits<double>::infinity();
        PopCd[FrontIdxs[1]] = std::numeric_limits<double>::infinity();
      } else {
        std::vector<std::vector<double>> Front;
        Front.reserve(FrontIdxs.size());
        for (auto Idx : FrontIdxs) {
          Front.push_back(Pop.f()[Idx]);
        }
        auto Cd = util::crowdingDistance(Front);
        for (decltype(Cd.size()) I = 0U; I < Cd.size(); ++I) {
          PopCd[FrontIdxs[I]] = Cd[I];
        }
      }
    }

    // We then loop thorugh all individuals with increment 4 to select two pairs
    // of parents that will each create 2 new offspring
    for (decltype(NP) I = 0U; I < NP; I += 4) {
      // We create two offsprings using the shuffled list 1
      Parent1Idx = util::moTournamentSelection(Shuffle1[I], Shuffle1[I + 1], Ndr, PopCd, Rng);
      Parent2Idx = util::moTournamentSelection(Shuffle1[I + 2], Shuffle1[I + 3], Ndr, PopCd, Rng);
      Children = util::sbxCrossover(Pop.x()[Parent1Idx], Pop.x()[Parent2Idx], Cr, Rng);
      util::polynomialMutation(Children.first, Bounds, M, Rng);
      util::polynomialMutation(Children.second, Bounds, M, Rng);

      Popnew.append(Children.first);
      Popnew.append(Children.second);

      // We repeat with the shuffled list 2
      Parent1Idx = util::moTournamentSelection(Shuffle2[I], Shuffle2[I + 1], Ndr, PopCd, Rng);
      Parent2Idx = util::moTournamentSelection(Shuffle2[I + 2], Shuffle2[I + 3], Ndr, PopCd, Rng);
      Children = util::sbxCrossover(Pop.x()[Parent1Idx], Pop.x()[Parent2Idx], Cr, Rng);
      util::polynomialMutation(Children.first, Bounds, M, Rng);
      util::polynomialMutation(Children.second, Bounds, M, Rng);

      Popnew.append(Children.first);
      Popnew.append(Children.second);
    }
    // Popnew now contains 2NP individuals

    // Save the best NP individuals in the population
    // according to the crowded comparison operator
    BestIdx = util::selectBestNMo(Popnew.f(), NP);
    for (decltype(NP) I = 0; I < NP; ++I) {
      Pop.insert(I, Popnew.x()[BestIdx[I]], Popnew.f()[BestIdx[I]]);
    }
  }

  return Pop;
}

} // namespace firestarter::optimizer::algorithm