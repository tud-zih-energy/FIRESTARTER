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

#pragma once

#include <cstring>
#include <firestarter/Measurement/Summary.hpp>
#include <firestarter/Optimizer/Individual.hpp>
#include <map>
#include <tuple>
#include <vector>

namespace firestarter::optimizer {

class Problem {
public:
  Problem() = default;
  virtual ~Problem() = default;

  // return the fitness for an individual
  virtual auto metrics(Individual const& Individual) -> std::map<std::string, firestarter::measurement::Summary> = 0;

  virtual auto fitness(std::map<std::string, firestarter::measurement::Summary> const& Summaries)
      -> std::vector<double> = 0;

  // get the bounds of the problem
  [[nodiscard]] virtual auto getBounds() const -> std::vector<std::tuple<unsigned, unsigned>> = 0;

  // get the number of dimensions of the problem
  [[nodiscard]] auto getDims() const -> std::size_t { return this->getBounds().size(); };

  // get the number of objectives.
  [[nodiscard]] virtual auto getNobjs() const -> std::size_t = 0;

  // is the problem multiobjective
  [[nodiscard]] auto isMO() const -> bool { return this->getNobjs() > 1; };

  // get the number of fitness evaluations
  [[nodiscard]] auto getFevals() const -> uint64_t { return Fevals; };

protected:
  // number of fitness evaluations
  uint64_t Fevals = 0;
};

} // namespace firestarter::optimizer
