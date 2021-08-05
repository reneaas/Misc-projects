#ifndef SOLARSYSTEM_HPP
#define SOLARSYSTEM_HPP

#include <armadillo>

#define pi 3.14159265359

class SolarSystem {
private:
    /* data */
    arma::cube pos_, vel_;
    arma::mat force_;
    arma::mat old_acc_, new_acc_;
    arma::vec mass_;

    int n_particles_, timesteps_;
    const int dims_ = 3;
    double dt_, dt2_;
    const double G = 4*pi*pi;

    void (SolarSystem::*advance_position)(int t);
    void (SolarSystem::*advance_velocity)(int t);
    void (SolarSystem::*acceleration)(int t, arma::mat *a);


    void euler_cromer_advance_position(int t);
    void euler_cromer_advance_velocity(int t);
    void verlet_advance_position(int t);
    void verlet_advance_velocity(int t);
    void acceleration_non_relativistic(int timestep, arma::mat *a);

public:
    SolarSystem(int n_particles, int timesteps, double step_sz);

    void init_data(arma::mat init_pos, arma::mat init_vel, arma::vec mass);
    void verlet();
    void euler_cromer();
    void write_to_file(std::string outfilename);
};

#endif
