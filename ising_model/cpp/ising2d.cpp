#include "ising2d.hpp"
#include <iostream>

Ising2D::Ising2D(int L, double T, std::string spin_config){
    L_ = L;
    n_spins_ = L*L;
    beta_ = 1./T;
    spin_config_ = spin_config;

    boltzmann_dist_ = arma::vec(17);
    for (int i = 0; i < 17; i+=4){
        boltzmann_dist_.at(i) = exp(-beta_*(i-8));
    }
    sampler = &Ising2D::metropolis;
}

/*
Implements the metropolis-hastings algorithm applied to the 2D ising model.
*/
void Ising2D::metropolis(SpinSystem *system){
    for (int n = 0; n < n_spins_; n++){
        int i = arma::randi(arma::distr_param(0, L_-1));
        int j = arma::randi(arma::distr_param(0, L_-1));
        int dE = 2*system->spin_mat(i, j)
                            * (system->spin_mat(i, j+1)
                            + system->spin_mat(i, j-1)
                            + system->spin_mat(i+1, j)
                            + system->spin_mat(i-1, j)
        );

        if (dE <= 0){
            system->energy_ += 1.*dE;
            system->spin_mat_.at(i, j) *= (-1);
            system->magnetization_ += 2*system->spin_mat_.at(i, j);
        }
        else if (arma::randu() < boltzmann_dist_.at(dE+8)){
            system->energy_ += 1.*dE;
            system->spin_mat_.at(i, j) *= (-1);
            system->magnetization_ += 2*system->spin_mat_.at(i, j);
        }
    }
}


arma::vec Ising2D::monte_carlo_sim(int mc_samples, int therm_samples){
    double E = 0, Esq = 0, M = 0, Msq = 0;
    #ifdef _OPENMP
    {
        #pragma omp parallel reduction(+:E, Esq, M, Msq)
        {
            arma::arma_rng::set_seed(omp_get_thread_num());
            SpinSystem system(L_, spin_config_);

            for (int i = 0; i < therm_samples; i++){
                (this->*sampler)(&system);
            }

            #pragma omp for
            for (int i = 0; i < mc_samples; i++){
                (this->*sampler)(&system);
                E += system.energy_;
                Esq += system.energy_*system.energy_;
                M += abs(system.magnetization_);
                Msq += system.magnetization_*system.magnetization_;
            }
        }
    }
    #else
    {
        arma::arma_rng::set_seed(10);
        SpinSystem system(L_, spin_config_);


        for (int i = 0; i < therm_samples; i++){
            (this->*sampler)(&system);
        }

        for (int i = 0; i < mc_samples; i++){
            (this->*sampler)(&system);
            E += system.energy_;
            Esq += system.energy_*system.energy_;
            M += abs(system.magnetization_);
            Msq += system.magnetization_*system.magnetization_;
        }
    }
    #endif

    E /= (mc_samples*n_spins_);
    Esq /= (mc_samples*n_spins_);
    M /= (mc_samples*n_spins_);
    Msq /= (mc_samples*n_spins_);

    arma::vec results = arma::vec(4);
    results.at(0) = E;
    results.at(1) = Esq;
    results.at(2) = M;
    results.at(3) = Msq;

    return results;
}
