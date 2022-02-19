#ifndef ISING2D_HPP
#define ISING2D_HPP


#include <armadillo>
#include <omp.h>

#include "spin_system.hpp"

class Ising2D {
private:
    arma::vec boltzmann_dist_;

    int L_, n_spins_, dE_;
    double beta_, wollf_acceptance_prob_;

    std::string spin_config_;

    //Pointers to member functions
    void (Ising2D::*sampler)(SpinSystem *system);


    //Sampling methods
    void metropolis(SpinSystem *system);


public:
    Ising2D(int L, double T, std::string spin_config);
    arma::vec monte_carlo_sim(int mc_samples, int therm_samples);
};

#endif
