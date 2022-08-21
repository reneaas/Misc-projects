#include "solar_system.hpp"

SolarSystem::SolarSystem() {}

SolarSystem::SolarSystem(
    std::vector<double> r0, 
    std::vector<double> v0,
    std::vector<double> m
) {
    r_ = r0;
    v_ = v0;
    m_ = m;
    G_ = 4 * PI * PI;
    num_objects_ = (int) m.size();
    dims_ = 3;
    force_.resize(r_.size());
}

std::vector<double> SolarSystem::get_position() {
    return r_;
}

std::vector<double> SolarSystem::get_velocity() {
    return v_;
}

std::vector<double> SolarSystem::get_mass() {
    return m_;
}

int SolarSystem::get_num_objects() {
    return num_objects_;
}

/* 
Uses Euler-Cromer to perform one integration step in time.
*/
void SolarSystem::step(std::vector<double> force, double dt) {

    for (int i = 0; i < num_objects_; i++) {
        v_[i * dims_ + 0] += dt * force[i * dims_ + 0] / m_[i];
        v_[i * dims_ + 1] += dt * force[i * dims_ + 1] / m_[i];
        v_[i * dims_ + 2] += dt * force[i * dims_ + 2] / m_[i];

        r_[i * dims_ + 0] += dt * v_[i * dims_ + 0];
        r_[i * dims_ + 1] += dt * v_[i * dims_ + 1];
        r_[i * dims_ + 2] += dt * v_[i * dims_ + 2];
    }
}

std::vector<double> SolarSystem::get_force() {
    for (int i = 0; i < num_objects_; i++) {
        std::vector<double> tmp = {0., 0., 0.}; 
        for (int j = 0; j < num_objects_; j++) {
            if (i != j) {
                double x_diff = r_[i * dims_ + 0] - r_[j * dims_ + 0];
                double y_diff = r_[i * dims_ + 1] - r_[j * dims_ + 1];
                double z_diff = r_[i * dims_ + 2] - r_[j * dims_ + 2];
                double r = (
                    x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
                );
                r = pow(r, 1.5);

                tmp[0] -= G_ * m_[i] * m_[j] * x_diff / r;
                tmp[1] -= G_ * m_[i] * m_[j] * y_diff / r;
                tmp[2] -= G_ * m_[i] * m_[j] * z_diff / r;
            }
        }
        force_[i * dims_ + 0] = tmp[0];
        force_[i * dims_ + 1] = tmp[1];
        force_[i * dims_ + 2] = tmp[2];

    }
    return force_;
}


