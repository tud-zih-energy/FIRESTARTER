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

using namespace firestarter::optimizer;

OptimizerWorker::OptimizerWorker(std::unique_ptr<firestarter::optimizer::Algorithm>&& algorithm,
                                 firestarter::optimizer::Population& population,
                                 std::string const& optimizationAlgorithm, unsigned individuals,
                                 std::chrono::seconds const& preheat)
    : Algorithm(std::move(algorithm))
    , Population(population)
    , OptimizationAlgorithm(optimizationAlgorithm)
    , Individuals(individuals)
    , Preheat(preheat) {
  pthread_create(&this->WorkerThread, NULL, reinterpret_cast<void* (*)(void*)>(OptimizerWorker::optimizerThread), this);
}

void OptimizerWorker::kill() {
  // we ignore ESRCH errno if thread already exited
  pthread_cancel(this->WorkerThread);
}

void OptimizerWorker::join() {
  // we ignore ESRCH errno if thread already exited
  pthread_join(this->WorkerThread, NULL);
}

void* OptimizerWorker::optimizerThread(void* optimizerWorker) {
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

  auto _this = reinterpret_cast<OptimizerWorker*>(optimizerWorker);

#ifndef __APPLE__
  pthread_setname_np(pthread_self(), "Optimizer");
#endif

  // heat the cpu before attempting to optimize
  std::this_thread::sleep_for(_this->Preheat);

  // For NSGA2 we start with a initial population
  if (_this->OptimizationAlgorithm == "NSGA2") {
    _this->Population.generateInitialPopulation(_this->Individuals);
  }

  _this->Algorithm->evolve(_this->Population);

  return NULL;
}
