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

#include "firestarter/Optimizer/Individual.hpp"

#include <random>
#include <utility>
#include <vector>

namespace firestarter::optimizer::util {

auto lessThanF(double A, double B) -> bool;

auto greaterThanF(double A, double B) -> bool;

auto paretoDominance(const std::vector<double>& Obj1, const std::vector<double>& Obj2) -> bool;

auto fastNonDominatedSorting(const std::vector<std::vector<double>>& Points)
    -> std::tuple<std::vector<std::vector<std::size_t>>, std::vector<std::vector<std::size_t>>,
                  std::vector<std::size_t>, std::vector<std::size_t>>;

auto crowdingDistance(const std::vector<std::vector<double>>& NonDomFront) -> std::vector<double>;

auto moTournamentSelection(std::vector<double>::size_type Idx1, std::vector<double>::size_type Idx2,
                           const std::vector<std::vector<double>::size_type>& NonDominationRank,
                           const std::vector<double>& CrowdingD, std::mt19937& Mt) -> std::vector<double>::size_type;

auto sbxCrossover(const firestarter::optimizer::Individual& Parent1, const firestarter::optimizer::Individual& Parent2,
                  double PCr, std::mt19937& Mt)
    -> std::pair<firestarter::optimizer::Individual, firestarter::optimizer::Individual>;

void polynomialMutation(firestarter::optimizer::Individual& Child,
                        const std::vector<std::tuple<unsigned, unsigned>>& Bounds, double PM, std::mt19937& Mt);

auto selectBestNMo(const std::vector<std::vector<double>>& InputF, std::size_t N) -> std::vector<std::size_t>;

auto ideal(const std::vector<std::vector<double>>& Points) -> std::vector<double>;

} // namespace firestarter::optimizer::util
