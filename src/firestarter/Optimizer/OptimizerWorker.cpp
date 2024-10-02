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

#include <firestarter/Optimizer/OptimizerWorker.hpp>

#include <thread>
#include <utility>

namespace firestarter::optimizer {

OptimizerWorker::OptimizerWorker(std::unique_ptr<firestarter::optimizer::Algorithm>&& Algorithm,
                                 firestarter::optimizer::Population& Population, std::string OptimizationAlgorithm,
                                 unsigned Individuals, std::chrono::seconds const& Preheat)
    : Algorithm(std::move(Algorithm))
    , Population(Population)
    , OptimizationAlgorithm(std::move(OptimizationAlgorithm))
    , Individuals(Individuals)
    , Preheat(Preheat) {
  pthread_create(&this->WorkerThread, nullptr, OptimizerWorker::optimizerThread, this);
}

void OptimizerWorker::kill() {
  // we ignore ESRCH errno if thread already exited
  pthread_cancel(WorkerThread);
}

void OptimizerWorker::join() {
  // we ignore ESRCH errno if thread already exited
  pthread_join(WorkerThread, nullptr);
}

auto OptimizerWorker::optimizerThread(void* OptimizerWorker) -> void* {
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);

  auto* This = reinterpret_cast<class OptimizerWorker*>(OptimizerWorker);

#ifndef __APPLE__
  pthread_setname_np(pthread_self(), "Optimizer");
#endif

  // heat the cpu before attempting to optimize
  std::this_thread::sleep_for(This->Preheat);

  // For NSGA2 we start with a initial population
  if (This->OptimizationAlgorithm == "NSGA2") {
    This->Population.generateInitialPopulation(This->Individuals);
  }

  This->Algorithm->evolve(This->Population);

  return nullptr;
}

} // namespace firestarter::optimizer