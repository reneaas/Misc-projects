#include "particle.hpp"

Particle::Particle(int n_particles, int dims){
    pos_ = arma::randn(dims, n_particles);
    trial_pos_ = arma::randn(dims, n_particles);
}

Particle::Particle(int n_particles, int dims, std::string sampling){
    arma::arma_rng::set_seed_random();
    // arma::arma_rng::set_seed(10);

    pos_ = arma::randn(dims, n_particles)*0.0001;
    trial_pos_ = arma::randn(dims, n_particles);

    if (sampling == "importance_sampling"){
        old_force_ = arma::randn(dims, n_particles);
        new_force_ = arma::randn(dims, n_particles);
    }
}
