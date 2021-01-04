/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2021 TU Dresden, Center for Information Services and High
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

#include <firestarter/Optimizer/Algorithm.hpp>
#include <firestarter/Optimizer/Population.hpp>

#include <chrono>
#include <memory>

extern "C" {
#include <pthread.h>
}

namespace firestarter::optimizer {

class OptimizerWorker {
public:
  OptimizerWorker(
      std::unique_ptr<firestarter::optimizer::Algorithm> &&algorithm,
      firestarter::optimizer::Population &population,
      std::string const &optimizationAlgorithm, unsigned individuals,
      std::chrono::seconds const &preheat);

  ~OptimizerWorker();

  void join();

private:
  static void *optimizerThread(void *optimizerWorker);

  std::unique_ptr<firestarter::optimizer::Algorithm> _algorithm;
  firestarter::optimizer::Population _population;
  std::string _optimizationAlgorithm;
  unsigned _individuals;
  std::chrono::seconds _preheat;

  pthread_t workerThread;
};

} // namespace firestarter::optimizer
