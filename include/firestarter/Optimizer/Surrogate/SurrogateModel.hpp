#pragma once

#include <armadillo>

namespace firestarter::optimizer::surrogate {

class SurrogateModel {
public:
  // train and select the fitting surrogate model
  SurrogateModel() {}

  virtual ~SurrogateModel() {}

  // get the name of the selected surrogate model
  std::string const &name() const { return _name; }

  // eval the selected surrogate model
  virtual double eval(arma::vec const &x) = 0;

protected:
  std::string _name;
};

} // namespace firestarter::optimizer::surrogate
