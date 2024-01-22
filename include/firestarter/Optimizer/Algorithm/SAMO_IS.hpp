#pragma once

#include <firestarter/Optimizer/Algorithm.hpp>

namespace firestarter::optimizer::algorithm {

class SAMO_IS : public Algorithm {
public:
  SAMO_IS(unsigned maxEvaluations, unsigned nsga2_individuals,
          unsigned nsga2_generations, double nsga2_cr, double nsga2_m);
  ~SAMO_IS() {}

  void checkPopulation(firestarter::optimizer::Population const &pop,
                       std::size_t populationSize) override;

  firestarter::optimizer::Population
  evolve(firestarter::optimizer::Population &pop) override;

private:
  unsigned _maxEvaluations;
  unsigned _nsga2_individuals;
  unsigned _nsga2_generations;
  double _nsga2_cr;
  double _nsga2_m;
};

} // namespace firestarter::optimizer::algorithm
