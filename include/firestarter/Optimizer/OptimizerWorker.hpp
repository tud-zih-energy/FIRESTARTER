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

#include "firestarter/Optimizer/Algorithm.hpp"
#include "firestarter/Optimizer/Population.hpp"
#include "firestarter/WindowsCompat.hpp" // IWYU pragma: keep

#include <chrono>
#include <memory>

namespace firestarter::optimizer {

/// Class to run the optimization in another thread.
class OptimizerWorker {
public:
  /// Start the optimization in another thread.
  /// \arg Algorithm The algorithm that is used to optimize FIRESTARTER.
  /// \arg Population The population containing the problem that will be used to optimize FIRESTARTER.
  /// \arg Individuals The number of individuals for the intial population.
  /// \arg Preheat The time we preheat before starting the optimization.
  OptimizerWorker(std::unique_ptr<firestarter::optimizer::Algorithm>&& Algorithm,
                  std::unique_ptr<firestarter::optimizer::Population>&& Population, unsigned Individuals,
                  std::chrono::seconds const& Preheat);

  ~OptimizerWorker() = default;

  /// Join the optimization thread.
  void join() const;

  /// Kill the optimization thread.
  void kill() const;

private:
  /// The thread worker that does the optimization.
  /// \arg OptimizerWorker The pointer to the OptimizerWorker (this) datastructure.
  /// \returns a nullptr
  static auto optimizerThread(void* OptimizerWorker) -> void*;

  /// The algorithm that is used to optimize FIRESTARTER.
  std::unique_ptr<firestarter::optimizer::Algorithm> Algorithm;
  /// The population containing the problem that will be used to optimize FIRESTARTER.
  std::unique_ptr<firestarter::optimizer::Population> Population;
  /// The number of individuals for the intial population.
  unsigned Individuals;
  /// The time we preheat before starting the optimization.
  std::chrono::seconds Preheat;

  /// The pthread that is used for the optimization.
  pthread_t WorkerThread{};
};

} // namespace firestarter::optimizer
