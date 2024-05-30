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

/********** Added Adiak and Caliper headers *********/
#define FIRESTARTER_WITH_CALIPER
#ifdef FIRESTARTER_WITH_CALIPER
#include <adiak.hpp>
#include <caliper/cali.h>
#endif

using namespace firestarter::optimizer;

OptimizerWorker::OptimizerWorker(
    std::unique_ptr<firestarter::optimizer::Algorithm> &&algorithm,
    firestarter::optimizer::Population &population,
    std::string const &optimizationAlgorithm, unsigned individuals,
    std::chrono::seconds const &preheat)
    : _algorithm(std::move(algorithm)), _population(population),
      _optimizationAlgorithm(optimizationAlgorithm), _individuals(individuals),
      _preheat(preheat) {
//#ifdef FIRESTARTER_WITH_CALIPER
//   CALI_MARK_BEGIN("pthread_create-optimizer-thread");
//#endif
   pthread_create(
      &this->workerThread, NULL,
      reinterpret_cast<void *(*)(void *)>(OptimizerWorker::optimizerThread),
      this);
//#ifdef FIRESTARTER_WITH_CALIPER
//   CALI_MARK_END("pthread_create-optimizer-thread");
//#endif
}

void OptimizerWorker::kill() {
  // we ignore ESRCH errno if thread already exited
  pthread_cancel(this->workerThread);
}

void OptimizerWorker::join() {
  // we ignore ESRCH errno if thread already exited
  pthread_join(this->workerThread, NULL);
}

void *OptimizerWorker::optimizerThread(void *optimizerWorker) {
#ifdef FIRESTARTER_WITH_CALIPER
   CALI_MARK_BEGIN("Optimizer-Thread");
#endif
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

  auto _this = reinterpret_cast<OptimizerWorker *>(optimizerWorker);

//#ifdef FIRESTARTER_WITH_CALIPER
//  CALI_MARK_BEGIN("pthread-optimizer");
//#endif
#ifndef __APPLE__
  pthread_setname_np(pthread_self(), "Optimizer");
#endif
//#ifdef FIRESTARTER_WITH_CALIPER
//  CALI_MARK_END("pthread-optimizer");
//#endif

  // heat the cpu before attempting to optimize
  std::this_thread::sleep_for(_this->_preheat);

  // For NSGA2 we start with a initial population
  if (_this->_optimizationAlgorithm == "NSGA2") {
	#ifdef FIRESTARTER_WITH_CALIPER
	  CALI_MARK_BEGIN("init-NSGA2");
	#endif
	_this->_population.generateInitialPopulation(_this->_individuals);
	#ifdef FIRESTARTER_WITH_CALIPER
	  CALI_MARK_END("init-NSGA2");
	#endif
  }

  _this->_algorithm->evolve(_this->_population);
#ifdef FIRESTARTER_WITH_CALIPER
  CALI_MARK_END("Optimizer-Thread");
#endif
  return NULL;
}
