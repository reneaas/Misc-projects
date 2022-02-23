#ifndef ISING2D_HPP
#define ISING2D_HPP

#include <random>

class Ising2D {
private:
    /* data */
    std::vector< std::vector<double> > spin_mat_;
    double beta_;
    std::vector<double> boltzmann_factor_;
    int L_;
    double E_, M_;

    mt19937_64 gen_; 
    uniform_int_distribution<int> uniform_int_dist_;
    uniform_real_distribution<double> uniform_real_dist_;

    void compute_init_observables();

public:
    Ising2D();
    Ising2D (std::vector< std::vector<double> > spin_matrix, double T);

    double spin_mat(int i, int j);
    double get_energy();
    double get_magnetization();
    std::vector< std::vector<double> > get_spin_matrix();


    virtual ~Ising2D (){};
};

#endif
