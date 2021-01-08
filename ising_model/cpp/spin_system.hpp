#ifndef SPIN_SYSTEM_HPP
#define SPIN_SYSTEM_HPP

#include <armadillo>


/*
SpinSystem is a class (could've been a struct, really) that stores the state of the spin system.
*/
class SpinSystem {
private:

public:
    SpinSystem(int L, std::string spin_config);

    arma::mat spin_mat_;
    int L_, n_spins_;
    double energy_, magnetization_;
};

#endif
