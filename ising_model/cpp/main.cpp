#include "ising2d.hpp"

#include <omp.h>
#include <iostream>

int main(int argc, char const *argv[]) {
    int L = 20;
    int mc_samples = 1e6;
    int therm_samples = 1e3;
    double T = 1.;
    std::string spin_config = "ordered";

    Ising2D my_solver(L, T, spin_config);
    double start = omp_get_wtime();
    my_solver.monte_carlo_sim(mc_samples, therm_samples);
    double end = omp_get_wtime();
    double timeused = end-start;

    std::cout << "timeused = " << timeused << " seconds " << std::endl;

    return 0;
}
