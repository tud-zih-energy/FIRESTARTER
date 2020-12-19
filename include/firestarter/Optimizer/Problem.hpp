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

#include <firestarter/Measurement/Summary.hpp>

#include <cstring>
#include <map>
#include <tuple>
#include <vector>

namespace firestarter::optimizer {

class Problem {
public:
  Problem() : _fevals(0) {}
  virtual ~Problem() {}

  // return the fitness for an individual
  virtual std::map<std::string, firestarter::measurement::Summary>
  metrics(std::vector<unsigned> const &individual) = 0;

  virtual std::vector<double>
  fitness(std::map<std::string, firestarter::measurement::Summary> const
              &summaries) = 0;

  // get the bounds of the problem
  virtual std::vector<std::tuple<unsigned, unsigned>> getBounds() const = 0;

  // get the number of dimensions of the problem
  std::size_t getDims() const { return this->getBounds().size(); };

  // get the number of objectives.
  virtual std::size_t getNobjs() const = 0;

  // is the problem multiobjective
  bool isMO() const { return this->getNobjs() > 1; };

  // get the number of fitness evaluations
  unsigned long long getFevals() const { return _fevals; };

protected:
  // number of fitness evaluations
  unsigned long long _fevals;
};

} // namespace firestarter::optimizer
