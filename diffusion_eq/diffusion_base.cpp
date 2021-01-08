#include "diffusion_base.hpp"

/*
Base class constructor.
*/
DiffusionBase::DiffusionBase(int n, double r){
    n_ = n;
    r_ = r;
    h_ = 1./((double) n+1);
    dt_ = r_*h_*h_;
    x_ = arma::linspace(0, 1, n_+1);
}
