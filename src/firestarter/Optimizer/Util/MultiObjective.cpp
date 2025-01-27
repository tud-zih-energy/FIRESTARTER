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

#include "firestarter/Optimizer/Util/MultiObjective.hpp"
#include "firestarter/Optimizer/Individual.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace firestarter::optimizer::util {

// Less than compares floating point types placing nans after inf or before -inf
// It is a useful function when calling e.g. std::sort to guarantee a weak
// strict ordering and avoid an undefined behaviour
auto lessThanF(double A, double B) -> bool {
  if (!std::isnan(A)) {
    if (!std::isnan(B)) {
      return A < B; // a < b
    }
    return true; // a < nan
  }
  // nan < b or nan < nan
  return false;
}

// Greater than compares floating point types placing nans after inf or before
// -inf It is a useful function when calling e.g. std::sort to guarantee a weak
// strict ordering and avoid an undefined behaviour
auto greaterThanF(double A, double B) -> bool {
  if (!std::isnan(A)) {
    if (!std::isnan(B)) {
      return A > B; // a > b
    }
    return false; // a > nan
  }
  // nan > b -> true
  // nan > nan -> false
  return !std::isnan(B);
}

/// Pareto-dominance
/**
 * Return true if \p obj1 Pareto dominates \p obj2, false otherwise.
 * Maximization is assumed.
 *
 * Each pair of corresponding elements in \p obj1 and \p obj2 is compared: if
 * all elements in \p obj1 are less or equal to the corresponding element in \p
 * obj2, but at least one is different, \p true will be returned. Otherwise, \p
 * false will be returned.
 *
 * @param obj1 first vector of objectives.
 * @param obj2 second vector of objectives.
 *
 * @return \p true if \p obj1 is dominating \p obj2, \p false otherwise.
 *
 * @throws std::invalid_argument if the dimensions of the two objectives are
 * different
 */
auto paretoDominance(const std::vector<double>& Obj1, const std::vector<double>& Obj2) -> bool {
  if (Obj1.size() != Obj2.size()) {
    throw std::invalid_argument(
        "Different number of objectives found in input fitnesses: " + std::to_string(Obj1.size()) + " and " +
        std::to_string(Obj2.size()) + ". I cannot define dominance");
  }
  bool FoundStrictlyDominatingDimension = false;
  for (decltype(Obj1.size()) I = 0U; I < Obj1.size(); ++I) {
    if (greaterThanF(Obj2[I], Obj1[I])) {
      return false;
    }
    if (lessThanF(Obj2[I], Obj1[I])) {
      FoundStrictlyDominatingDimension = true;
    }
  }
  return FoundStrictlyDominatingDimension;
}

/// Fast non dominated sorting
/**
 * An implementation of the fast non dominated sorting algorithm. Complexity is
 * \f$ O(MN^2)\f$ where \f$M\f$ is the number of objectives and \f$N\f$ is the
 * number of individuals.
 *
 * See: Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic
 * algorithm for multi-objective optimization: NSGA-II." Parallel problem
 * solving from nature PPSN VI. Springer Berlin Heidelberg, 2000.
 *
 * @param points An std::vector containing the objectives of different
 * individuals. Example
 * {{1,2,3},{-2,3,7},{-1,-2,-3},{0,0,0}}
 *
 * @return an std::tuple containing:
 *  - the non dominated fronts, an
 * <tt>std::vector<std::vector<std::size_t>></tt> containing the non dominated
 * fronts. Example {{1,2},{3},{0}}
 *  - the domination list, an <tt>std::vector<std::vector<std::size_t>></tt>
 * containing the domination list, i.e. the indexes of all individuals
 * dominated by the individual at position \f$i\f$. Example {{},{},{0,3},{0}}
 *  - the domination count, an <tt>std::vector<std::size_t></tt> containing the
 * number of individuals that dominate the individual at position \f$i\f$.
 * Example {2, 0, 0, 1}
 *  - the non domination rank, an <tt>std::vector<std::size_t></tt> containing
 * the index of the non dominated front to which the individual at position
 * \f$i\f$ belongs. Example {2,0,0,1}
 *
 * @throws std::invalid_argument If the size of \p points is not at least 2
 */
auto fastNonDominatedSorting(const std::vector<std::vector<double>>& Points)
    -> std::tuple<std::vector<std::vector<std::size_t>>, std::vector<std::vector<std::size_t>>,
                  std::vector<std::size_t>, std::vector<std::size_t>> {
  auto N = Points.size();
  // We make sure to have two points at least (one could also be allowed)
  if (N < 2U) {
    throw std::invalid_argument("At least two points are needed for fast_non_dominated_sorting: " + std::to_string(N) +
                                " detected.");
  }
  // Initialize the return values
  std::vector<std::vector<std::size_t>> NonDomFronts(1U);
  std::vector<std::vector<std::size_t>> DomList(N);
  std::vector<std::size_t> DomCount(N);
  std::vector<std::size_t> NonDomRank(N);

  // Start the fast non dominated sort algorithm
  for (decltype(N) I = 0U; I < N; ++I) {
    DomList[I].clear();
    DomCount[I] = 0U;
    for (decltype(N) J = 0U; J < I; ++J) {
      if (paretoDominance(Points[I], Points[J])) {
        DomList[I].push_back(J);
        ++DomCount[J];
      } else if (paretoDominance(Points[J], Points[I])) {
        DomList[J].push_back(I);
        ++DomCount[I];
      }
    }
  }
  for (decltype(N) I = 0U; I < N; ++I) {
    if (DomCount[I] == 0U) {
      NonDomRank[I] = 0U;
      NonDomFronts[0].push_back(I);
    }
  }
  // we copy dom_count as we want to output its value at this point
  auto DomCountCopy(DomCount);
  auto CurrentFront = NonDomFronts[0];
  std::vector<std::vector<std::size_t>>::size_type FrontCounter(0U);
  while (!CurrentFront.empty()) {
    std::vector<std::size_t> NextFront;
    for (const auto& P : CurrentFront) {
      for (const auto& Q : DomList[P]) {
        --DomCountCopy[Q];
        if (DomCountCopy[Q] == 0U) {
          NonDomRank[Q] = FrontCounter + 1U;
          NextFront.push_back(Q);
        }
      }
    }
    ++FrontCounter;
    CurrentFront = NextFront;
    if (!CurrentFront.empty()) {
      NonDomFronts.push_back(CurrentFront);
    }
  }
  return std::make_tuple(std::move(NonDomFronts), std::move(DomList), std::move(DomCount), std::move(NonDomRank));
}

/// Crowding distance
/**
 * An implementation of the crowding distance. Complexity is \f$ O(MNlog(N))\f$
 * where \f$M\f$ is the number of objectives and \f$N\f$ is the number of
 * individuals. The function assumes the input is a non-dominated front.
 * Failiure to this condition will result in undefined behaviour.
 *
 * See: Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic
 * algorithm for multi-objective optimization: NSGA-II." Parallel problem
 * solving from nature PPSN VI. Springer Berlin Heidelberg, 2000.
 *
 * @param non_dom_front An <tt>std::vector<std::vector<double>></tt> containing
 * a non dominated front. Example
 * {{0,0},{-1,1},{2,-2}}
 *
 * @returns a std::vector<double> containing the crowding distances. Example:
 * {2, inf, inf}
 *
 * @throws std::invalid_argument If \p non_dom_front does not contain at least
 * two points
 * @throws std::invalid_argument If points in \p do not all have at least two
 * objectives
 * @throws std::invalid_argument If points in \p non_dom_front do not all have
 * the same dimensionality
 */
auto crowdingDistance(const std::vector<std::vector<double>>& NonDomFront) -> std::vector<double> {
  auto N = NonDomFront.size();
  // We make sure to have two points at least
  if (N < 2U) {
    throw std::invalid_argument("A non dominated front must contain at least two points: " + std::to_string(N) +
                                " detected.");
  }
  auto M = NonDomFront[0].size();
  // We make sure the first point of the input non dominated front contains at
  // least two objectives
  if (M < 2U) {
    throw std::invalid_argument("Points in the non dominated front must "
                                "contain at least two objectives: " +
                                std::to_string(M) + " detected.");
  }
  // We make sure all points contain the same number of objectives
  if (!std::all_of(NonDomFront.begin(), NonDomFront.end(),
                   [M](const std::vector<double>& Item) { return Item.size() == M; })) {
    throw std::invalid_argument("A non dominated front must contain points of "
                                "uniform dimensionality. Some "
                                "different sizes were instead detected.");
  }
  std::vector<std::size_t> Indexes(N);
  std::iota(Indexes.begin(), Indexes.end(), static_cast<std::size_t>(0U));
  std::vector<double> Retval(N, 0.);
  for (decltype(M) I = 0U; I < M; ++I) {
    std::sort(Indexes.begin(), Indexes.end(), [I, &NonDomFront](std::size_t Idx1, std::size_t Idx2) {
      return lessThanF(NonDomFront[Idx1][I], NonDomFront[Idx2][I]);
    });
    Retval[Indexes[0]] = std::numeric_limits<double>::infinity();
    Retval[Indexes[N - 1U]] = std::numeric_limits<double>::infinity();
    const double Df = NonDomFront[Indexes[N - 1U]][I] - NonDomFront[Indexes[0]][I];
    for (decltype(N - 2U) J = 1U; J < N - 1U; ++J) {
      Retval[Indexes[J]] += (NonDomFront[Indexes[J + 1U]][I] - NonDomFront[Indexes[J - 1U]][I]) / Df;
    }
  }
  return Retval;
}

// Multi-objective tournament selection. Requires all sizes to be consistent.
// Does not check if input is well formed.
auto moTournamentSelection(std::vector<double>::size_type Idx1, std::vector<double>::size_type Idx2,
                           const std::vector<std::vector<double>::size_type>& NonDominationRank,
                           const std::vector<double>& CrowdingD, std::mt19937& Mt) -> std::vector<double>::size_type {
  if (NonDominationRank[Idx1] < NonDominationRank[Idx2]) {
    return Idx1;
  }
  if (NonDominationRank[Idx1] > NonDominationRank[Idx2]) {
    return Idx2;
  }
  if (CrowdingD[Idx1] > CrowdingD[Idx2]) {
    return Idx1;
  }
  if (CrowdingD[Idx1] < CrowdingD[Idx2]) {
    return Idx2;
  }
  std::uniform_real_distribution<> Drng(0., 1.);
  return ((Drng(Mt) < 0.5) ? Idx1 : Idx2);
}

// Implementation of the binary crossover.
// Requires the crossover probability p_cr  in[0,1] -> undefined algo behaviour
// otherwise Requires dimensions of the parent and bounds to be equal -> out of
// bound reads. nix is the integer dimension (integer alleles assumed at the end
// of the chromosome)
auto sbxCrossover(const firestarter::optimizer::Individual& Parent1, const firestarter::optimizer::Individual& Parent2,
                  const double PCr, std::mt19937& Mt)
    -> std::pair<firestarter::optimizer::Individual, firestarter::optimizer::Individual> {
  // Decision vector dimensions
  auto Nix = Parent1.size();
  // Initialize the child decision vectors
  firestarter::optimizer::Individual Child1 = Parent1;
  firestarter::optimizer::Individual Child2 = Parent2;
  // Random distributions
  std::uniform_real_distribution<> Drng(0.,
                                        1.); // to generate a number in [0, 1)

  // This implements a Simulated Binary Crossover SBX
  if (Drng(Mt) < PCr) { // No crossever at all will happen with probability p_cr
    // This implements two-points crossover and applies it to the integer part
    // of the chromosome.
    if (Nix > 0U) {
      std::uniform_int_distribution<firestarter::optimizer::Individual::size_type> RaNum(0, Nix - 1U);
      auto Site1 = RaNum(Mt);
      auto Site2 = RaNum(Mt);
      if (Site1 > Site2) {
        std::swap(Site1, Site2);
      }
      for (decltype(Site2) J = Site1; J <= Site2; ++J) {
        Child1[J] = Parent2[J];
        Child2[J] = Parent1[J];
      }
    }
  }
  return std::make_pair(std::move(Child1), std::move(Child2));
}

// Performs polynomial mutation. Requires all sizes to be consistent. Does not
// check if input is well formed. p_m is the mutation probability
void polynomialMutation(firestarter::optimizer::Individual& Child,
                        const std::vector<std::tuple<unsigned, unsigned>>& Bounds, const double PM, std::mt19937& Mt) {
  // Decision vector dimensions
  auto Nix = Child.size();
  // Random distributions
  std::uniform_real_distribution<> Drng(0.,
                                        1.); // to generate a number in [0, 1)
  // This implements the integer mutation for an individual
  for (decltype(Nix) J = 0; J < Nix; ++J) {
    if (Drng(Mt) < PM) {
      // We need to draw a random integer in [lb, ub].
      auto Lb = std::get<0>(Bounds[J]);
      auto Ub = std::get<1>(Bounds[J]);
      std::uniform_int_distribution<firestarter::optimizer::Individual::size_type> Dist(Lb, Ub);
      auto Mutated = Dist(Mt);
      Child[J] = Mutated;
    }
  }
}

/// Selects the best N individuals in multi-objective optimization
/**
 * Selects the best N individuals out of a population, (intended here as an
 * <tt>std::vector<std::vector<double>></tt> containing the  objective vectors).
 * The strict ordering used is the same as that defined in
 * pagmo::sort_population_mo.
 *
 * Complexity is \f$ O(MN^2)\f$ where \f$M\f$ is the number of objectives and
 * \f$N\f$ is the number of individuals.
 *
 * While the complexity is the same as that of pagmo::sort_population_mo, this
 * function returns a permutation of:
 *
 * @code{.unparsed}
 * auto ret = pagmo::sort_population_mo(input_f).resize(N);
 * @endcode
 *
 * but it is faster than the above code: it avoids to compute the crowidng
 * distance for all individuals and only computes it for the last non-dominated
 * front that contains individuals included in the best N.
 *
 * If N is zero, an empty vector will be returned.
 *
 * @param input_f Input objectives vectors. Example {{0.25,0.25},{-1,1},{2,-2}};
 * @param N Number of best individuals to return
 *
 * @returns an <tt>std::vector</tt> containing the indexes of the best N
 * objective vectors. Example {2,1}
 *
 * @throws unspecified all exceptions thrown by
 * pagmo::fast_non_dominated_sorting and pagmo::crowding_distance
 */
auto selectBestNMo(const std::vector<std::vector<double>>& InputF, std::size_t N) -> std::vector<std::size_t> {
  if (N == 0U) { // corner case
    return {};
  }
  if (InputF.empty()) { // corner case
    return {};
  }
  if (InputF.size() == 1U) { // corner case
    return {0U};
  }
  if (N >= InputF.size()) { // corner case
    std::vector<std::size_t> Retval(InputF.size());
    std::iota(Retval.begin(), Retval.end(), static_cast<std::size_t>(0U));
    return Retval;
  }
  std::vector<std::size_t> Retval;
  std::vector<std::size_t>::size_type FrontId(0U);
  // Run fast-non-dominated sorting
  auto Tuple = fastNonDominatedSorting(InputF);
  // Insert all non dominated fronts if not more than N
  for (const auto& Front : std::get<0>(Tuple)) {
    if (Retval.size() + Front.size() <= N) {
      for (auto I : Front) {
        Retval.push_back(I);
      }
      if (Retval.size() == N) {
        return Retval;
      }
      ++FrontId;
    } else {
      break;
    }
  }
  auto Front = std::get<0>(Tuple)[FrontId];
  std::vector<std::vector<double>> NonDomFits(Front.size());
  // Run crowding distance for the front
  for (decltype(Front.size()) I = 0U; I < Front.size(); ++I) {
    NonDomFits[I] = InputF[Front[I]];
  }
  std::vector<double> Cds(crowdingDistance(NonDomFits));
  // We now have front and crowding distance, we sort the front w.r.t. the
  // crowding
  std::vector<std::size_t> Idxs(Front.size());
  std::iota(Idxs.begin(), Idxs.end(), static_cast<std::size_t>(0U));
  std::sort(Idxs.begin(), Idxs.end(), [&Cds](std::size_t Idx1, std::size_t Idx2) {
    return greaterThanF(Cds[Idx1], Cds[Idx2]);
  }); // Descending order1
  auto Remaining = N - Retval.size();
  for (decltype(Remaining) I = 0U; I < Remaining; ++I) {
    Retval.push_back(Front[Idxs[I]]);
  }
  return Retval;
}

/// Ideal point
/**
 * Computes the ideal point of an input population, (intended here as an
 * <tt>std::vector<std::vector<double>></tt> containing the  objective vectors).
 *
 * Complexity is \f$ O(MN)\f$ where \f$M\f$ is the number of objectives and
 * \f$N\f$ is the number of individuals.
 *
 * @param points Input objectives vectors. Example
 * {{-1,3,597},{1,2,3645},{2,9,789},{0,0,231},{6,-2,4576}};
 *
 * @returns A std::vector<double> containing the ideal point. Example:
 * {6,9,4576}
 *
 * @throws std::invalid_argument if the input objective vectors are not all of
 * the same size
 */
auto ideal(const std::vector<std::vector<double>>& Points) -> std::vector<double> {
  // Corner case
  if (Points.empty()) {
    return {};
  }

  // Sanity checks
  auto M = Points[0].size();
  for (const auto& F : Points) {
    if (F.size() != M) {
      throw std::invalid_argument("Input vector of objectives must contain "
                                  "fitness vector of equal dimension " +
                                  std::to_string(M));
    }
  }
  // Actual algorithm
  std::vector<double> Retval(M);
  for (decltype(M) I = 0U; I < M; ++I) {
    Retval[I] = (*std::min_element(Points.begin(), Points.end(),
                                   [I](const std::vector<double>& F1, const std::vector<double>& F2) {
                                     return util::greaterThanF(F1[I], F2[I]);
                                   }))[I];
  }
  return Retval;
}

} // namespace firestarter::optimizer::util
