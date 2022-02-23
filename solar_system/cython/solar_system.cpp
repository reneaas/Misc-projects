#include "solar_system.hpp"

SolarSystem::SolarSystem(){}

SolarSystem::SolarSystem(
    std::vector< std::vector<double> > r0, 
    std::vector< std::vector<double> > v0,
    std::vector<double> m
) {
    r_ = r0;
    v_ = v0;
    m_ = m;
    G_ = 4 * PI * PI;
}

std::vector< std::vector<double> > SolarSystem::get_position() {
    return r_;
}

std::vector< std::vector<double> > SolarSystem::get_velocity() {
    return v_;
}

std::vector<double> SolarSystem::get_mass() {
    return m_;
}

int SolarSystem::get_num_objects() {
    return r_.size();
}

void SolarSystem::step(std::vector< std::vector<double> > force, double dt) {
    std::vector< std::vector<double> > new_r, new_v;
    for (std::size_t i = 0; i < r_.size(); i++) {
        v_[i][0] = v_[i][0] + dt * force[i][0] / m_[i];
        v_[i][1] = v_[i][1] + dt * force[i][1] / m_[i];
        v_[i][2] = v_[i][2] + dt * force[i][2] / m_[i];


        r_[i][0] = r_[i][0] + dt * v_[i][0];
        r_[i][1] = r_[i][1] + dt * v_[i][1];
        r_[i][2] = r_[i][2] + dt * v_[i][2];

    }
}

std::vector< std::vector<double> > SolarSystem::get_force() {
    std::vector< std::vector<double> > force;
    for (std::size_t i = 0; i < r_.size(); i++) {
        std::vector<double> tmp; 
        tmp.resize(3);
        for (std::size_t j = 0; j < r_.size(); j++) {
            if (i != j) {
                double x_diff = r_[i][0] - r_[j][0];
                double y_diff = r_[i][1] - r_[j][1];
                double z_diff = r_[i][2] - r_[j][2];
                double r = (
                    x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
                );
                r = pow(r, 1.5);

                tmp[0] -= G_ * m_[i] * m_[j] * x_diff / r;
                tmp[1] -= G_ * m_[i] * m_[j] * y_diff / r;
                tmp[2] -= G_ * m_[i] * m_[j] * z_diff / r;
            }
        }
        force.push_back(tmp);
    }
    return force;
}

std::vector< std::vector< std::vector<double> > > SolarSystem::compute_evolution(int num_timesteps, double dt) {
    std::vector< std::vector< std::vector<double> > > R;
    R.resize(num_timesteps);

    for (int i = 0; i < num_timesteps; i++) {
        std::vector< std::vector<double> > force = get_force();
        step(force, dt);
        R[i] = r_;
        // R.push_back(r_);
    }
    return R;
}
