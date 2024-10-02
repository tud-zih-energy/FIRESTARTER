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

#include "../Algorithm.hpp"

namespace firestarter::optimizer::algorithm {

class NSGA2 : public Algorithm {
public:
  NSGA2(unsigned Gen, double Cr, double M);
  ~NSGA2() override = default;

  void checkPopulation(firestarter::optimizer::Population const& Pop, std::size_t PopulationSize) override;

  auto evolve(firestarter::optimizer::Population& Pop) -> firestarter::optimizer::Population override;

private:
  const unsigned Gen;
  const double Cr;
  const double M;
};

} // namespace firestarter::optimizer::algorithm
