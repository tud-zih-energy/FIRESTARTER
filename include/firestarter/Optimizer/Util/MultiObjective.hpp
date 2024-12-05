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

#pragma once

#include <firestarter/Optimizer/Individual.hpp>

#include <random>
#include <utility>
#include <vector>

namespace firestarter::optimizer::util {

bool less_than_f(double a, double b);

bool greater_than_f(double a, double b);

bool pareto_dominance(const std::vector<double> &obj1,
                      const std::vector<double> &obj2);

std::tuple<std::vector<std::vector<std::size_t>>,
           std::vector<std::vector<std::size_t>>, std::vector<std::size_t>,
           std::vector<std::size_t>>
fast_non_dominated_sorting(const std::vector<std::vector<double>> &points);

std::vector<double>
crowding_distance(const std::vector<std::vector<double>> &non_dom_front);

std::vector<double>::size_type mo_tournament_selection(
    std::vector<double>::size_type idx1, std::vector<double>::size_type idx2,
    const std::vector<std::vector<double>::size_type> &non_domination_rank,
    const std::vector<double> &crowding_d, std::mt19937 &mt);

std::pair<firestarter::optimizer::Individual,
          firestarter::optimizer::Individual>
sbx_crossover(const firestarter::optimizer::Individual &parent1,
              const firestarter::optimizer::Individual &parent2,
              const double p_cr, std::mt19937 &mt);

void polynomial_mutation(
    firestarter::optimizer::Individual &child,
    const std::vector<std::tuple<unsigned, unsigned>> &bounds, const double p_m,
    std::mt19937 &mt);

std::vector<std::size_t>
select_best_N_mo(const std::vector<std::vector<double>> &input_f,
                 std::size_t N);

std::vector<double> ideal(const std::vector<std::vector<double>> &points);

} // namespace firestarter::optimizer::util
