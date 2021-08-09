#ifndef SPIN_SYSTEM_HPP
#define SPIN_SYSTEM_HPP

#include <armadillo>


/*
SpinSystem is a class (could've been a struct, really) that stores the state of the spin system.
*/
class SpinSystem {
private:
    //Helper functions
    void init_observables();
    int idx(int index);
public:
    SpinSystem(int L, std::string spin_config);
    int spin_mat(int i, int j);
    void add_to_cluster(int i, int j, double acceptance_prob);


    arma::imat spin_mat_;
    arma::imat cluster_;
    int L_, n_spins_;
    double energy_, magnetization_;
    double wollf_acceptance_prob_;
};

#endif
