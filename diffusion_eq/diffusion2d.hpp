#ifndef DIFFUSION2D_HPP
#define DIFFUSION2D_HPP

#include <armadillo>
#include <omp.h>
#include "diffusion_base.hpp"

class Diffusion2D  : public DiffusionBase {
private:
    arma::mat v_old_, v_new_;
    arma::vec y_;
    int m_;


    //Helper functions
    double advance(int i, int j);


public:
    Diffusion2D(int n, double r);
    void init_cond(double f(double x, double y));
    void solve(double tot_time);
};

#endif
