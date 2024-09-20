#pragma once

#include <firestarter/Optimizer/Problem.hpp>
#include <firestarter/Optimizer/Surrogate/SurrogateSelector.hpp>

#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <thread>
#include <tuple>
#include <utility>

namespace firestarter::optimizer::problem {

class SurrogateProblem final : public firestarter::optimizer::Problem {
public:
  SurrogateProblem(std::vector<std::string> const &metrics,
                   std::vector<std::tuple<unsigned, unsigned>> const &bounds)
      : _metrics(metrics), _bounds(bounds) {
    assert(_metrics.size() != 0);

    auto const &hist_x = History::x();
    auto const dims = hist_x[0].size();

    arma::vec boundsLow(dims);
    arma::vec boundsUp(dims);
    arma::mat x(dims, hist_x.size());

    {
      for (std::size_t i = 0; i < dims; ++i) {
        boundsLow(i) = std::get<0>(bounds[i]);
        boundsUp(i) = std::get<1>(bounds[i]);
      }

      for (std::size_t i = 0; i < hist_x.size(); ++i) {
        x.col(i) = arma::conv_to<arma::vec>::from(hist_x[i]);
      }
    }

    for (auto const &metric : metrics) {
      std::string strippedMetricName;
      arma::vec y(hist_x.size());

      if (metric[0] == '-') {
        strippedMetricName = metric.substr(1);
      } else {
        strippedMetricName = metric;
      }

      for (std::size_t i = 0; i < hist_x.size(); ++i) {
        // fill y with 3 digits precision
        y(i) = (double)((int)(History::find(hist_x[i])
                                  .value()[strippedMetricName]
                                  .average *
                              1000)) /
               1000.0;
      }

      auto model = std::make_unique<
          firestarter::optimizer::surrogate::SurrogateSelector>(boundsLow,
                                                                boundsUp, x, y);
      log::info() << "Using surrogate model " << model->name() << " for metric "
                  << metric;
      _models.push_back(std::move(model));
    }
  }

  ~SurrogateProblem() {}

  // return all available metrics for the individual
  std::map<std::string, firestarter::measurement::Summary>
  metrics(std::vector<unsigned> const &individual) override {
    std::map<std::string, firestarter::measurement::Summary> metrics = {};
    for (std::size_t i = 0; i < _metrics.size(); ++i) {
      auto name = _metrics[i];
      auto value = _models[i]->eval(arma::conv_to<arma::vec>::from(individual));
      firestarter::measurement::Summary summary;
      summary.average = value;
      metrics[name] = summary;
    }
    return metrics;
  }

  std::vector<double> fitness(
      std::map<std::string, firestarter::measurement::Summary> const &summaries)
      override {
    std::vector<double> values = {};

    for (auto const &metricName : _metrics) {
      auto findName = [metricName](auto const &summary) {
        auto invertedName = "-" + summary.first;
        return metricName.compare(summary.first) == 0 ||
               metricName.compare(invertedName) == 0;
      };

      auto it = std::find_if(summaries.begin(), summaries.end(), findName);

      if (it == summaries.end()) {
        continue;
      }

      // round to two decimal places after the comma
      auto value = std::round(it->second.average * 100.0) / 100.0;

      // invert metric
      if (metricName[0] == '-') {
        value *= -1.0;
      }

      values.push_back(value);
    }

    return values;
  }

  // get the bounds of the problem
  std::vector<std::tuple<unsigned, unsigned>> getBounds() const override {
    return _bounds;
  }

  // get the number of objectives.
  std::size_t getNobjs() const override { return _metrics.size(); }

  std::vector<std::string> metrics() const override { return _metrics; }

private:
  std::vector<std::string> _metrics;
  std::vector<std::tuple<unsigned, unsigned>> _bounds;
  std::vector<
      std::unique_ptr<firestarter::optimizer::surrogate::SurrogateSelector>>
      _models;
};

} // namespace firestarter::optimizer::problem
