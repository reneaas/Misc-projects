#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <armadillo>

class Particle {

public:
    int dims_, n_particles_;
    arma::mat pos_, trial_pos_;
    arma::mat old_force_, new_force_;

    Particle(int n_particles, int dims);
    Particle(int n_particles, int dims, std::string sampling);
};

#endif
