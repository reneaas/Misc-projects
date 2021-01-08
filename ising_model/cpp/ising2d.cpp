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

    idx_ = arma::vec(L_+2);
    idx_.at(0) = L_-1;
    idx_.at(L_+1) = 0;
    for (int i = 1; i < L_+1; i++){
        idx_.at(i) = i-1;
    }
}


void Ising2D::init_observables(SpinSystem* system){
    for (int j = 1; j <= L_; j++){
        for (int i = 1; i <= L_; i++){
            system->energy_ -= system->spin_mat_.at(idx_.at(i), idx_.at(j))
                        *(system->spin_mat_.at(idx_.at(i+1), idx_.at(j))
                        + system->spin_mat_.at(idx_.at(i), idx_.at(j+1)));

            system->magnetization_ += system->spin_mat_.at(idx_.at(i), idx_.at(j));
        }
    }
}

void Ising2D::metropolis(int i, int j, SpinSystem *system){
    int dE = 2*system->spin_mat_.at(idx_.at(i), idx_.at(j))
                *(system->spin_mat_.at(idx_.at(i+1), idx_.at(j)) + system->spin_mat_.at(idx_.at(i-1), idx_.at(j))
                + system->spin_mat_.at(idx_.at(i), idx_.at(j+1)) + system->spin_mat_.at(idx_.at(i), idx_.at(j-1)));

    if (dE <= 0){
        system->energy_ += 1.*dE;
        system->spin_mat_.at(idx_.at(i), idx_.at(j)) *= (-1);
        system->magnetization_ += 2*system->spin_mat_.at(idx_.at(i), idx_.at(j));
    }
    else if (arma::randu() < boltzmann_dist_.at(dE+8)){
        system->energy_ += 1.*dE;
        system->spin_mat_.at(idx_.at(i), idx_.at(j)) *= (-1);
        system->magnetization_ += 2*system->spin_mat_.at(idx_.at(i), idx_.at(j));
    }
}

void Ising2D::monte_carlo_sim(int mc_samples, int therm_samples){
    // std::uniform_int_distribution<int> dist_int(1, L_);
    arma::vec energies = arma::vec(mc_samples);
    #ifdef _OPENMP
    {
        #pragma omp parallel
        {
            std::mt19937_64 gen(omp_get_thread_num() + 42);
            std::uniform_int_distribution<int> dist_int(1, L_);
            SpinSystem system(L_, spin_config_);
            init_observables(&system);


            for (int i = 0; i < therm_samples*n_spins_; i++){
                int x = dist_int(gen);
                int y = dist_int(gen);
                metropolis(x, y, &system);
            }

            #pragma omp for
            for (int i = 0; i < mc_samples; i++){
                for (int n = 0; n < n_spins_; n++){
                    int x = dist_int(gen);
                    int y = dist_int(gen);
                    metropolis(x, y, &system);
                }
                energies.at(i) = system.energy_;
            }
        }
    }
    #else
    {
        std::mt19937_64 gen(42);
        std::uniform_int_distribution<int> dist_int(1, L_);
        SpinSystem system(L_, spin_config_);
        init_observables(&system);

        for (int i = 0; i < therm_samples*n_spins_; i++){
            int x = dist_int(gen_);
            int y = dist_int(gen_);
            metropolis(x, y, &system);
        }

        for (int i = 0; i < mc_samples; i++){
            for (int n = 0; n < n_spins_; n++){
                int x = dist_int(gen_);
                int y = dist_int(gen_);
                metropolis(x, y, &system);
            }
            energies.at(i) = energy_;
        }
    }
    #endif


    double mean_energy = arma::mean(energies);
    double stddev = arma::stddev(energies);

    std::cout << "<E> = " << mean_energy/n_spins_ << std::endl;
}
