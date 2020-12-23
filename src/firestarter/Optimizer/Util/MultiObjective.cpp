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

// This file borrows a lot of code from https://github.com/esa/pagmo2

#include <firestarter/Optimizer/Util/MultiObjective.hpp>

#include <algorithm>
#include <stdexcept>

namespace firestarter::optimizer::util {

// Less than compares floating point types placing nans after inf or before -inf
// It is a useful function when calling e.g. std::sort to guarantee a weak
// strict ordering and avoid an undefined behaviour
bool less_than_f(double a, double b) {
  if (!std::isnan(a)) {
    if (!std::isnan(b))
      return a < b; // a < b
    else
      return true; // a < nan
  } else {
    if (!std::isnan(b))
      return false; // nan < b
    else
      return false; // nan < nan
  }
}

// Greater than compares floating point types placing nans after inf or before
// -inf It is a useful function when calling e.g. std::sort to guarantee a weak
// strict ordering and avoid an undefined behaviour
bool greater_than_f(double a, double b) {
  if (!std::isnan(a)) {
    if (!std::isnan(b))
      return a > b; // a > b
    else
      return false; // a > nan
  } else {
    if (!std::isnan(b))
      return true; // nan > b
    else
      return false; // nan > nan
  }
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
bool pareto_dominance(const std::vector<double> &obj1,
                      const std::vector<double> &obj2) {
  if (obj1.size() != obj2.size()) {
    throw std::invalid_argument(
        "Different number of objectives found in input fitnesses: " +
        std::to_string(obj1.size()) + " and " + std::to_string(obj2.size()) +
        ". I cannot define dominance");
  }
  bool found_strictly_dominating_dimension = false;
  for (decltype(obj1.size()) i = 0u; i < obj1.size(); ++i) {
    if (greater_than_f(obj2[i], obj1[i])) {
      return false;
    } else if (less_than_f(obj2[i], obj1[i])) {
      found_strictly_dominating_dimension = true;
    }
  }
  return found_strictly_dominating_dimension;
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
std::tuple<std::vector<std::vector<std::size_t>>,
           std::vector<std::vector<std::size_t>>, std::vector<std::size_t>,
           std::vector<std::size_t>>
fast_non_dominated_sorting(const std::vector<std::vector<double>> &points) {
  auto N = points.size();
  // We make sure to have two points at least (one could also be allowed)
  if (N < 2u) {
    throw std::invalid_argument(
        "At least two points are needed for fast_non_dominated_sorting: " +
        std::to_string(N) + " detected.");
  }
  // Initialize the return values
  std::vector<std::vector<std::size_t>> non_dom_fronts(1u);
  std::vector<std::vector<std::size_t>> dom_list(N);
  std::vector<std::size_t> dom_count(N);
  std::vector<std::size_t> non_dom_rank(N);

  // Start the fast non dominated sort algorithm
  for (decltype(N) i = 0u; i < N; ++i) {
    dom_list[i].clear();
    dom_count[i] = 0u;
    for (decltype(N) j = 0u; j < i; ++j) {
      if (pareto_dominance(points[i], points[j])) {
        dom_list[i].push_back(j);
        ++dom_count[j];
      } else if (pareto_dominance(points[j], points[i])) {
        dom_list[j].push_back(i);
        ++dom_count[i];
      }
    }
  }
  for (decltype(N) i = 0u; i < N; ++i) {
    if (dom_count[i] == 0u) {
      non_dom_rank[i] = 0u;
      non_dom_fronts[0].push_back(i);
    }
  }
  // we copy dom_count as we want to output its value at this point
  auto dom_count_copy(dom_count);
  auto current_front = non_dom_fronts[0];
  std::vector<std::vector<std::size_t>>::size_type front_counter(0u);
  while (current_front.size() != 0u) {
    std::vector<std::size_t> next_front;
    for (decltype(current_front.size()) p = 0u; p < current_front.size(); ++p) {
      for (decltype(dom_list[current_front[p]].size()) q = 0u;
           q < dom_list[current_front[p]].size(); ++q) {
        --dom_count_copy[dom_list[current_front[p]][q]];
        if (dom_count_copy[dom_list[current_front[p]][q]] == 0u) {
          non_dom_rank[dom_list[current_front[p]][q]] = front_counter + 1u;
          next_front.push_back(dom_list[current_front[p]][q]);
        }
      }
    }
    ++front_counter;
    current_front = next_front;
    if (current_front.size() != 0u) {
      non_dom_fronts.push_back(current_front);
    }
  }
  return std::make_tuple(std::move(non_dom_fronts), std::move(dom_list),
                         std::move(dom_count), std::move(non_dom_rank));
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
std::vector<double>
crowding_distance(const std::vector<std::vector<double>> &non_dom_front) {
  auto N = non_dom_front.size();
  // We make sure to have two points at least
  if (N < 2u) {
    throw std::invalid_argument(
        "A non dominated front must contain at least two points: " +
        std::to_string(N) + " detected.");
  }
  auto M = non_dom_front[0].size();
  // We make sure the first point of the input non dominated front contains at
  // least two objectives
  if (M < 2u) {
    throw std::invalid_argument("Points in the non dominated front must "
                                "contain at least two objectives: " +
                                std::to_string(M) + " detected.");
  }
  // We make sure all points contain the same number of objectives
  if (!std::all_of(
          non_dom_front.begin(), non_dom_front.end(),
          [M](const std::vector<double> &item) { return item.size() == M; })) {
    throw std::invalid_argument("A non dominated front must contain points of "
                                "uniform dimensionality. Some "
                                "different sizes were instead detected.");
  }
  std::vector<std::size_t> indexes(N);
  std::iota(indexes.begin(), indexes.end(), std::size_t(0u));
  std::vector<double> retval(N, 0.);
  for (decltype(M) i = 0u; i < M; ++i) {
    std::sort(indexes.begin(), indexes.end(),
              [i, &non_dom_front](std::size_t idx1, std::size_t idx2) {
                return less_than_f(non_dom_front[idx1][i],
                                   non_dom_front[idx2][i]);
              });
    retval[indexes[0]] = std::numeric_limits<double>::infinity();
    retval[indexes[N - 1u]] = std::numeric_limits<double>::infinity();
    double df =
        non_dom_front[indexes[N - 1u]][i] - non_dom_front[indexes[0]][i];
    for (decltype(N - 2u) j = 1u; j < N - 1u; ++j) {
      retval[indexes[j]] += (non_dom_front[indexes[j + 1u]][i] -
                             non_dom_front[indexes[j - 1u]][i]) /
                            df;
    }
  }
  return retval;
}

// Multi-objective tournament selection. Requires all sizes to be consistent.
// Does not check if input is well formed.
std::vector<double>::size_type mo_tournament_selection(
    std::vector<double>::size_type idx1, std::vector<double>::size_type idx2,
    const std::vector<std::vector<double>::size_type> &non_domination_rank,
    const std::vector<double> &crowding_d, std::mt19937 &mt) {
  if (non_domination_rank[idx1] < non_domination_rank[idx2])
    return idx1;
  if (non_domination_rank[idx1] > non_domination_rank[idx2])
    return idx2;
  if (crowding_d[idx1] > crowding_d[idx2])
    return idx1;
  if (crowding_d[idx1] < crowding_d[idx2])
    return idx2;
  std::uniform_real_distribution<> drng(0., 1.);
  return ((drng(mt) < 0.5) ? idx1 : idx2);
}

// Implementation of the binary crossover.
// Requires the crossover probability p_cr  in[0,1] -> undefined algo behaviour
// otherwise Requires dimensions of the parent and bounds to be equal -> out of
// bound reads. nix is the integer dimension (integer alleles assumed at the end
// of the chromosome)
std::pair<firestarter::optimizer::Individual,
          firestarter::optimizer::Individual>
sbx_crossover(const firestarter::optimizer::Individual &parent1,
              const firestarter::optimizer::Individual &parent2,
              const double p_cr, std::mt19937 &mt) {
  // Decision vector dimensions
  auto nix = parent1.size();
  firestarter::optimizer::Individual::size_type site1, site2;
  // Initialize the child decision vectors
  firestarter::optimizer::Individual child1 = parent1;
  firestarter::optimizer::Individual child2 = parent2;
  // Random distributions
  std::uniform_real_distribution<> drng(0.,
                                        1.); // to generate a number in [0, 1)

  // This implements a Simulated Binary Crossover SBX
  if (drng(mt) <
      p_cr) { // No crossever at all will happen with probability p_cr
    // This implements two-points crossover and applies it to the integer part
    // of the chromosome.
    if (nix > 0u) {
      std::uniform_int_distribution<
          firestarter::optimizer::Individual::size_type>
          ra_num(0, nix - 1u);
      site1 = ra_num(mt);
      site2 = ra_num(mt);
      if (site1 > site2) {
        std::swap(site1, site2);
      }
      for (decltype(site2) j = site1; j <= site2; ++j) {
        child1[j] = parent2[j];
        child2[j] = parent1[j];
      }
    }
  }
  return std::make_pair(std::move(child1), std::move(child2));
}

// Performs polynomial mutation. Requires all sizes to be consistent. Does not
// check if input is well formed. p_m is the mutation probability
void polynomial_mutation(
    firestarter::optimizer::Individual &child,
    const std::vector<std::tuple<unsigned, unsigned>> &bounds, const double p_m,
    std::mt19937 &mt) {
  // Decision vector dimensions
  auto nix = child.size();
  // Random distributions
  std::uniform_real_distribution<> drng(0.,
                                        1.); // to generate a number in [0, 1)
  // This implements the integer mutation for an individual
  for (decltype(nix) j = 0; j < nix; ++j) {
    if (drng(mt) < p_m) {
      // We need to draw a random integer in [lb, ub].
      auto lb = std::get<0>(bounds[j]);
      auto ub = std::get<1>(bounds[j]);
      std::uniform_int_distribution<
          firestarter::optimizer::Individual::size_type>
          dist(lb, ub);
      auto mutated = dist(mt);
      child[j] = mutated;
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
std::vector<std::size_t>
select_best_N_mo(const std::vector<std::vector<double>> &input_f,
                 std::size_t N) {
  if (N == 0u) { // corner case
    return {};
  }
  if (input_f.size() == 0u) { // corner case
    return {};
  }
  if (input_f.size() == 1u) { // corner case
    return {0u};
  }
  if (N >= input_f.size()) { // corner case
    std::vector<std::size_t> retval(input_f.size());
    std::iota(retval.begin(), retval.end(), std::size_t(0u));
    return retval;
  }
  std::vector<std::size_t> retval;
  std::vector<std::size_t>::size_type front_id(0u);
  // Run fast-non-dominated sorting
  auto tuple = fast_non_dominated_sorting(input_f);
  // Insert all non dominated fronts if not more than N
  for (const auto &front : std::get<0>(tuple)) {
    if (retval.size() + front.size() <= N) {
      for (auto i : front) {
        retval.push_back(i);
      }
      if (retval.size() == N) {
        return retval;
      }
      ++front_id;
    } else {
      break;
    }
  }
  auto front = std::get<0>(tuple)[front_id];
  std::vector<std::vector<double>> non_dom_fits(front.size());
  // Run crowding distance for the front
  for (decltype(front.size()) i = 0u; i < front.size(); ++i) {
    non_dom_fits[i] = input_f[front[i]];
  }
  std::vector<double> cds(crowding_distance(non_dom_fits));
  // We now have front and crowding distance, we sort the front w.r.t. the
  // crowding
  std::vector<std::size_t> idxs(front.size());
  std::iota(idxs.begin(), idxs.end(), std::size_t(0u));
  std::sort(idxs.begin(), idxs.end(),
            [&cds](std::size_t idx1, std::size_t idx2) {
              return greater_than_f(cds[idx1], cds[idx2]);
            }); // Descending order1
  auto remaining = N - retval.size();
  for (decltype(remaining) i = 0u; i < remaining; ++i) {
    retval.push_back(front[idxs[i]]);
  }
  return retval;
}

} // namespace firestarter::optimizer::util
