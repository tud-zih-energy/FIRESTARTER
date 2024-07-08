#pragma once

#include <firestarter/Optimizer/Surrogate/Model/SOT.hpp>
#include <firestarter/Optimizer/Surrogate/SurrogateModel.hpp>

#include <armadillo>

#define REGISTER(NAME...)                                                      \
  [](arma::vec const &boundsLow, arma::vec const &boundsUp,                    \
     arma::mat const &x, arma::vec const &y) -> SurrogateModel * {             \
    return new model::NAME(boundsLow, boundsUp, x, y);                         \
  }

namespace firestarter::optimizer::surrogate {

class SurrogateSelector {
public:
  // train and select the fitting surrogate model
  SurrogateSelector(arma::vec const &boundsLow, arma::vec const &boundsUp,
                    arma::mat const &x, arma::vec const &y) {
    arma::mat x_train, x_eval;
    arma::vec y_train, y_eval_true;

    // split the dataset into train and validation
    arma::vec train_idx;
    arma::vec eval_idx;
    {
      // random shuffle of indicies from 0 to y.n_elem - 1
      arma::uvec shuffle = arma::randperm(y.n_elem);
      // get first 90% for train and last 10% for eval
      auto split_point = std::ceil((double)shuffle.n_elem * 0.9);

      arma::uvec train_idx = shuffle.head(split_point);
      arma::uvec eval_idx = shuffle.tail(shuffle.n_elem - split_point);

      if (eval_idx.n_elem == 0) {
        assert((false, "No elements left for evaluation of surrogate models in "
                       "__FILE__ __LINE__"));
      }

      x_train = x.cols(train_idx);
      x_eval = x.cols(eval_idx);
      y_train = y(train_idx);
      y_eval_true = y(eval_idx);
    }

    // train the models
    std::vector<std::unique_ptr<SurrogateModel>> models;
    for (auto const &ctor : _modelsCtor) {
      models.push_back(std::move(std::unique_ptr<SurrogateModel>(
          ctor(boundsLow, boundsUp, x_train, y_train))));
    }

    // vector of mean squared error
    arma::vec mse(models.size());

    for (std::size_t i = 0; i < models.size(); ++i) {
      auto &model = models[i];
      arma::vec y_eval(y_eval_true.n_elem);

      for (std::size_t j = 0; j < x_eval.n_cols; ++j) {
        arma::vec x = x_eval.col(j);
        y_eval(j) = model->eval(x);
      }

      arma::vec diff = y_eval - y_eval_true;
      mse(i) = sum(diff % diff) / (double)y_eval.n_elem;

      log::info() << "mean squared error of " << model->name() << " = "
                  << mse(i);
    }

    auto idx = arma::index_min(mse);

    _model = std::move(models[idx]);
  }

  ~SurrogateSelector() {}

  // get the name of the selected surrogate model
  std::string const &name() const { return _model->name(); }

  // eval the selected surrogate model
  double eval(arma::vec const &x) { return _model->eval(x); }

private:
  // list of surrogate ctors
  const std::vector<std::function<SurrogateModel *(
      arma::vec const &boundsLow, arma::vec const &boundsUp, arma::mat const &x,
      arma::vec const &y)>>
      _modelsCtor = {
          REGISTER(SOT<sot::CubicRBF>), REGISTER(SOT<sot::TpsRBF>),
          REGISTER(
              SOT<sot::RBFInterpolant<sot::LinearKernel, sot::ConstantTail>>)};

#undef REGISTER

  std::unique_ptr<SurrogateModel> _model;
};

} // namespace firestarter::optimizer::surrogate
