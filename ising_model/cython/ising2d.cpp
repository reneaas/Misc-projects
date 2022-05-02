#include "ising2d.hpp"

using namespace std;

Ising2D::Ising2D(){}

Ising2D::Ising2D(std::vector< std::vector<double> > spin_matrix, double T) {
    spin_mat_ = spin_matrix;
    beta_ = 1. / T;
    boltzmann_factor_ = std::vector<double>(17, 0.0);
    for (int i = -8; i < 9; i+=4) {
        boltzmann_factor_[i + 8] = exp(-beta_ * 1. * i);
    }
    L_ = spin_matrix.size();
    compute_init_observables();

    uniform_int_dist_ = uniform_int_distribution<int>(0, L_);
}

void Ising2D::compute_init_observables() {
    E_ = 0;
    M_ = 0;
    for (int i = 0; i < L_; i++) {
        for (int j = 0; j < L_; j++) {
            M_ += spin_mat(i, j);
            E_ += spin_mat(i, j) * (spin_mat(i+1, j) + spin_mat(i, j + 1));
        }
    }
}

double Ising2D::spin_mat(int i, int j) {
    return spin_mat_[(i + L_) % L_][(j + L_) % L_];
}

double Ising2D::get_energy() {
    return E_;
}

double Ising2D::get_magnetization() {
    return M_;
}

std::vector< std::vector<double> > Ising2D::get_spin_matrix() {
    return spin_mat_;
}

void Ising2D::metropolis() {
    for (int n = 0; n < L_ * L_; n++) {
        int i = uniform_int_dist_(gen_);
        int j = uniform_int_dist_(gen_);

        double dE = 2 * spin_mat(i, j) * (
            spin_mat(i+1, j) + spin_mat(i-1, j)
            + spin_mat(i, j+1) + spin_mat(i, j-1)
        );

        if (dE < 0) {
            E_ += dE;
            spin_mat_[i][j] *= (-1);
            M_ += 2 * spin_mat_[i][j];
        }
        else if (boltzmann_factor_[(int) dE] < uniform_real_dist_()) {
            E_ += dE;
            spin_mat_[i][j] *= (-1);
            M_ += 2 * spin_mat_[i][j];
        }
    }
}

std::vector<double> run_chain (int num_burnin_steps, int num_results) {
    for (int i = 0; i < num_burnin_steps; i++) {
        metropolis();
    }

    std::vector<double> energies;
    energies.resize(num_results)
    for (int i = 0; i < num_results; i++) {
        metropolis();
        
    }
}