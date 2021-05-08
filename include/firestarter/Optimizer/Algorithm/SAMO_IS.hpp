#pragma once

#include <firestarter/Optimizer/Algorithm.hpp>

namespace firestarter::optimizer::algorithm {

class SAMO_IS : public Algorithm {
public:
  SAMO_IS(unsigned maxEvaluations, double cr, double m);
  ~SAMO_IS() {}

  void checkPopulation(firestarter::optimizer::Population const &pop,
                       std::size_t populationSize) override;

  firestarter::optimizer::Population
  evolve(firestarter::optimizer::Population &pop) override;

private:
  unsigned _maxEvaluations;
  double _cr;
  double _m;
};

} // namespace firestarter::optimizer::algorithm
