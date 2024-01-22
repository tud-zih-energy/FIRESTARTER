#include <firestarter/Logging/Log.hpp>
#include <firestarter/Optimizer/Algorithm/NSGA2.hpp>
#include <firestarter/Optimizer/Algorithm/SAMO_IS.hpp>
#include <firestarter/Optimizer/Individual.hpp>
#include <firestarter/Optimizer/Problem/SurrogateProblem.hpp>
#include <firestarter/Optimizer/Util/MultiObjective.hpp>

#include <armadillo>

#include <algorithm>
#include <set>
#include <stdexcept>

using namespace firestarter::optimizer::algorithm;

SAMO_IS::SAMO_IS(unsigned maxEvaluations, unsigned nsga2_individuals,
                 unsigned nsga2_generations, double nsga2_cr, double nsga2_m)
    : Algorithm(), _maxEvaluations(maxEvaluations),
      _nsga2_individuals(nsga2_individuals),
      _nsga2_generations(nsga2_generations), _nsga2_cr(nsga2_cr),
      _nsga2_m(nsga2_m) {
  if (nsga2_cr >= 1. || nsga2_cr < 0.) {
    throw std::invalid_argument("The crossover probability must be in the "
                                "[0,1[ range, while a value of " +
                                std::to_string(nsga2_cr) + " was detected");
  }
  if (nsga2_m < 0. || nsga2_m > 1.) {
    throw std::invalid_argument("The mutation probability must be in the [0,1] "
                                "range, while a value of " +
                                std::to_string(nsga2_m) + " was detected");
  }
}

void SAMO_IS::checkPopulation(firestarter::optimizer::Population const &pop,
                              std::size_t populationSize) {
  const auto &prob = pop.problem();

  if (!prob.isMO()) {
    throw std::invalid_argument("SAMO-IS is a multiobjective algorithms, while "
                                "number of objectives is " +
                                std::to_string(prob.getNobjs()));
  }

  auto minPopulationSize = 2 * (prob.getDims() + 1);

  if (populationSize < minPopulationSize) {
    throw std::invalid_argument(
        "for SAMO-IS the population size must be "
        "greater equal 2 times the number of (variables plus one)."
        "The population size must be greater equal " +
        std::to_string(minPopulationSize) + ", while it is " +
        std::to_string(populationSize));
  }
}

firestarter::optimizer::Population
SAMO_IS::evolve(firestarter::optimizer::Population &pop) {
  const auto &prob = pop.problem();
  const auto bounds = prob.getBounds();
  auto NP = pop.size();
  auto fevals0 = prob.getFevals();

  this->checkPopulation(
      const_cast<firestarter::optimizer::Population const &>(pop), NP);

  std::random_device rd;
  std::mt19937 rng(rd());

  std::pair<Individual, Individual> children;

  for (unsigned gen = 1u; History::x().size() < _maxEvaluations; ++gen) {

    {
      std::stringstream ss;

      ss << std::endl
         << std::setw(7) << "SAMO-IS Gen:" << std::setw(15) << "Fevals:";
      for (decltype(prob.getNobjs()) i = 0; i < prob.getNobjs(); ++i) {
        ss << std::setw(15) << "ideal" << std::to_string(i + 1u) << ":";
      }
      ss << std::endl;

      // Print the logs
      std::vector<double> idealPoint = util::ideal(pop.f());

      ss << std::setw(7 + 8) << gen << std::setw(15)
         << prob.getFevals() - fevals0;
      for (decltype(idealPoint.size()) i = 0; i < idealPoint.size(); ++i) {
        ss << std::setw(15) << idealPoint[i];
      }

      firestarter::log::info() << ss.str();
    }

    // At each generation we make a copy of the population into popnew
    firestarter::optimizer::Population popnew(pop);

    // run nsga-ii with the surrogate and the limit for 20 generations with an
    // population size of 100
    auto surrogateProblem =
        std::make_shared<firestarter::optimizer::problem::SurrogateProblem>(
            prob.metrics(), bounds);
    firestarter::optimizer::Population surrogatePopulation(
        std::move(surrogateProblem), false);

    surrogatePopulation.generateInitialPopulation(_nsga2_individuals);
    NSGA2 nsga2(_nsga2_generations, _nsga2_cr, _nsga2_m);

    auto solutions = nsga2.evolve(surrogatePopulation);

    auto fnds_res = util::fast_non_dominated_sorting(solutions.f());
    auto ndf = std::get<0>(fnds_res);

    std::vector<Individual> C_ND_x(pop.x());
    std::vector<std::vector<double>> C_ND_f(pop.f());

    // add the non dominated solutions to C_NDset (_x and _f)
    for (auto const &idx : ndf[0]) {
      C_ND_x.push_back(solutions.x()[idx]);
      C_ND_f.push_back(solutions.f()[idx]);
    }

    {

      auto max_num_solutions = std::min(NP, C_ND_x.size());

      auto best_idx = util::select_best_N_mo(C_ND_f, max_num_solutions);

      // only add new solutions
      for (auto const &idx : best_idx) {
        if (!History::find(C_ND_x[idx]).has_value()) {
          popnew.append(C_ND_x[idx]);
        }
      }
    }

    {
      auto best_idx = util::select_best_N_mo(popnew.f(), NP);
      // We insert into the population
      for (decltype(NP) i = 0;
           (i < NP) && (History::x().size() < _maxEvaluations); ++i) {
        pop.insert(i, popnew.x()[best_idx[i]], popnew.f()[best_idx[i]]);
      }
    }
  }

  return pop;
}
