#pragma once

#include <firestarter/Logging/Log.hpp>
#include <firestarter/Optimizer/Surrogate/SurrogateModel.hpp>

#include <sot.h>

namespace firestarter::optimizer::surrogate::model {

template <class T> class SOT : public SurrogateModel {
  static_assert(std::is_base_of<sot::Surrogate, T>::value,
                "T must extend sot::Surrogate");

public:
  // train and select the fitting surrogate model
  SOT(arma::vec const &boundsLow, arma::vec const &boundsUp, arma::mat const &x,
      arma::vec const &y) {
    assert(boundsLow.n_elem == boundsUp.n_elem);
    assert(boundsLow.n_elem == x.n_rows);
    assert(x.n_cols == y.n_elem);

    _name = typeid(T).name();
    _model =
        std::make_unique<T>(y.n_elem, boundsLow.n_elem, boundsLow, boundsUp);

    _model->addPoints(x, y);

    _model->fit();
  }

  ~SOT() {}

  // eval the selected surrogate model
  double eval(arma::vec const &x) override { return _model->eval(x); }

private:
  std::unique_ptr<T> _model;
};

} // namespace firestarter::optimizer::surrogate::model
