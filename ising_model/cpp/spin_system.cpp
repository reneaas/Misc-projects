#include "spin_system.hpp"


SpinSystem::SpinSystem(int L, std::string spin_config){
    if (spin_config == "ordered"){
        spin_mat_ = arma::mat(L, L).fill(1.);
    }
    else if (spin_config == "random"){
        spin_mat_ = 2*arma::randi<arma::mat>(L, L, arma::distr_param(0,1)) - 1;
        // spin_mat_ = 2*arma::randi(L, L, arma::distr_param(0,1))-1;
    }

    energy_ = 0.;
    magnetization_ = 0.;
}
