#include "solarsystem.hpp"

SolarSystem::SolarSystem(int n_particles, int timesteps, double step_sz){
    n_particles_ = n_particles;
    dt_ = step_sz;
    dt2_ = 0.5*dt_*dt_;
    timesteps_ = timesteps;

    pos_ = arma::cube(dims_, n_particles_, timesteps_).fill(0.);
    vel_ = arma::cube(dims_, n_particles_, timesteps_).fill(0.);
    force_ = arma::mat(dims_, n_particles_).fill(0.);
    old_acc_ = arma::mat(dims_, n_particles_).fill(0.);
    new_acc_ = arma::mat(dims_, n_particles_).fill(0.);

    mass_ = arma::vec(n_particles_);

    acceleration = &SolarSystem::acceleration_non_relativistic;
}

void SolarSystem::init_data(arma::mat init_pos, arma::mat init_vel, arma::vec mass){
    mass_ = mass;
    pos_.slice(0) = init_pos;
    vel_.slice(0) = init_vel;
}

void SolarSystem::verlet(){
    advance_position = &SolarSystem::verlet_advance_position;
    advance_velocity = &SolarSystem::verlet_advance_velocity;

    (this->*acceleration)(0., &old_acc_);
    for (int t = 0; t < timesteps_-1; t++){
        (this->*advance_position)(t);
        (this->*acceleration)(t+1, &new_acc_);
        (this->*advance_velocity)(t);
        old_acc_.swap(new_acc_);
    }
}

void SolarSystem::euler_cromer(){
    advance_position = &SolarSystem::euler_cromer_advance_position;
    advance_velocity = &SolarSystem::euler_cromer_advance_velocity;

    for (int t = 0; t < timesteps_-1; t++){
        (this->*acceleration)(t, &old_acc_);
        (this->*advance_velocity)(t);
        (this->*advance_position)(t);
    }
}

void SolarSystem::euler_cromer_advance_position(int t){
    pos_.slice(t+1) = pos_.slice(t) + vel_.slice(t+1)*dt_;
}

void SolarSystem::euler_cromer_advance_velocity(int t){
    vel_.slice(t+1) = vel_.slice(t) + old_acc_*dt_;
}

void SolarSystem::verlet_advance_position(int t){
    pos_.slice(t+1) = pos_.slice(t) + vel_.slice(t)*dt_ + dt2_*old_acc_;
}

void SolarSystem::verlet_advance_velocity(int t){
    vel_.slice(t+1) = vel_.slice(t) + 0.5*dt_*(old_acc_ + new_acc_);
}

void SolarSystem::acceleration_non_relativistic(int timestep, arma::mat *a){
    arma::mat pos = pos_.slice(timestep);
    arma::vec diff, ri;
    double r_ij;
    (*a).fill(0.);
    for (int i = 0; i < n_particles_; i++){
        for (int j = i+1; j < n_particles_; j++){
            diff = pos.col(i) - pos.col(j);
            r_ij = arma::norm(diff);
            r_ij = pow(r_ij, 3);
            diff /= r_ij;
            (*a).col(i) -= mass_.at(j)*diff;
            (*a).col(j) += mass_.at(i)*diff; //Newtons third law
        }
    }
    (*a) *= G;
}

void SolarSystem::write_to_file(std::string outfilename){
    std::ofstream ofile;
    ofile.open(outfilename);
    for (int i = 0; i < timesteps_; i++){
        arma::mat r_mat = pos_.slice(i);
        for (int j = 0; j < n_particles_; j++){
            ofile << r_mat.at(0, j) << " " << r_mat.at(1, j) << " " << r_mat.at(2, j) << std::endl;
        }
    }
    ofile.close();
}
