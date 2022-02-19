#include "spin_system.hpp"


SpinSystem::SpinSystem(int L, std::string spin_config){
    L_ = L;
    n_spins_ = L*L;
    if (spin_config == "ordered"){
        spin_mat_ = arma::imat(L, L).fill(1);
    }
    else if (spin_config == "random"){
        spin_mat_ = 2*arma::randi<arma::imat>(L, L, arma::distr_param(0,1)) - 1;
    }

    init_observables();
}

/*
Helper function to compute the initial conditions of the system.
*/
void SpinSystem::init_observables(){
    energy_ = 0.;
    magnetization_ = 0.;
    for (int i = 0; i < L_; i++){
        for (int j = 0; j < L_; j++){
            energy_ -= spin_mat(i, j)
                                *(spin_mat(i, j+1)
                                + spin_mat(i+1, j)
            );

            magnetization_ += spin_mat(i, j);
        }
    }
}

/*
Helper function to provide easier access to matrix elements
with periodic boundary conditions imposed automatically.
*/
int SpinSystem::spin_mat(int i, int j){
    return spin_mat_.at(idx(i), idx(j));
}

/*
Implements periodic boundary conditions
*/
int SpinSystem::idx(int index){
    return (index + L_) % L_;
}
