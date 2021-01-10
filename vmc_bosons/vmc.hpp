#ifndef VMC_HPP
#define VMC_HPP

#include <armadillo>
#include <iostream>
#include "particle.hpp"
#include <fstream>
#include <omp.h>

class VMC {
private:

    int n_particles_, dims_;
    double step_sz_, alpha_, omega_, sqrt_step_sz_, beta_, hard_core_diam_, a_;

    double E_mean_, EE_mean_;
    arma::vec energies_, radii_;

    std::string sampling_;

    //Variables for importance sampling stuff
    double D_, D_inv_;



    //Various quantum forces
    void quantum_force_spherical(arma::mat *pos, arma::mat *force);
    void quantum_force_elliptical(arma::mat *pos, arma::mat *force);


    //Pointers to member functions
    void (VMC::*gen_trial_pos)(Particle *particle);
    double (VMC::*loc_energy)(Particle *particle);
    double (VMC::*trial_fn)(Particle *particle);
    void (VMC::*metropolis)(Particle *particle, double *last_trial, double *energy);
    void (VMC::*quantum_force)(arma::mat *pos, arma::mat *force);

    //Brute force sampling
    void gen_trial_pos_bf(Particle *particle);
    void metropolis_bf(Particle *particle, double *last_trial, double *energy);

    //Importance sampling
    void gen_trial_pos_is(Particle *particle);
    void metropolis_is(Particle *particle, double *last_trial, double *energy);
    double greens_fn(arma::mat x, arma::mat y, arma::mat force_y);

    //No interacting Bosons
    double trial_fn_no_int(Particle *particle);
    double loc_energy_no_int(Particle *particle);

    //Interacting Bosons
    double loc_energy_with_int(Particle *particle);
    double trial_fn_with_int(Particle *particle);


    void metropolis_one_body_density(Particle *particle, double *last_trial, double *r);



public:

    VMC(int n_particles, int dims, double step_sz, double alpha, std::string sampling);
    VMC(int n_particles, int dims, double step_sz, double alpha, double beta, std::string sampling);
    // VMC(int n_particles, double step_sz, double alpha, std::string sampling);
    double monte_carlo_sim(int mc_samples, int therm_samples);
    void one_body_density(int mc_samples, int therm_samples, std::string filename);
    void write_to_file(std::string filename);
    void write_to_file_all(std::string filename);

    //Statistical analysis
    void bootstrap(double *mean_energy, double *stddev, int bootstrap_samples);
    void blocking(double *mean_energy, double *stddev, int blocking_transforms);

    void test();
};

#endif
