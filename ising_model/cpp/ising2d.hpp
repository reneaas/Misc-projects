#ifndef ISING2D_HPP
#define ISING2D_HPP


#include <armadillo>
#include <omp.h>

#include "spin_system.hpp"

class Ising2D {
private:
    arma::vec boltzmann_dist_, idx_;

    int L_, n_spins_, dE_;
    double beta_;

    std::mt19937_64 gen_;
    // std::uniform_real_distribution<double> uniform_rng_;

    std::string spin_config_;

    void init_observables(SpinSystem *system);
    void metropolis(int i, int j, SpinSystem *system);


public:
    Ising2D(int L, double T, std::string spin_config);
    void monte_carlo_sim(int mc_samples, int therm_samples);
};

#endif
