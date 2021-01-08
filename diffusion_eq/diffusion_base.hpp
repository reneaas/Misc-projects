#ifndef DIFFUSIONBASE_HPP
#define DIFFUSIONBASE_HPP


#include <armadillo>

class DiffusionBase {
protected:
    /* data */
    int n_;
    double r_, dt_, h_;
    arma::vec x_;

public:
    DiffusionBase(int n, double r);
};

#endif
