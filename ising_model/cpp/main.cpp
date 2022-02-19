#include "ising2d.hpp"
#include <iostream>
#include <omp.h>

int main(int argc, char const *argv[]) {
    int L = 40;
    int mc_samples = 1e4;
    int therm_samples = 1e2;
    double T = 1.;
    std::string spin_config = "ordered";

    Ising2D my_solver(L, T, spin_config);
    double start = omp_get_wtime();
    arma::vec results = my_solver.monte_carlo_sim(mc_samples, therm_samples);
    double end = omp_get_wtime();
    double timeused = end-start;

    results.print("results = ");
    std::cout << "timeused = " << timeused << " seconds " << std::endl;

    return 0;
}
