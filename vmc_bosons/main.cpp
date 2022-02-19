#include "vmc.hpp"

#include <armadillo>
#include <iostream>
#include <omp.h>

using namespace std;

void optimize(double alpha0, int max_iter, int n_particles);

int main(int argc, char const *argv[]) {
    //int n_particles = std::atoi(argv[1]);
    int n_particles = 50;
    int dims = 3;
    double step_sz = 0.1;
    double alpha = 0.5;
    int mc_samples = pow(2,19);
    int therm_samples = 10000;
    int n = 50;
    int bootstrap_samples = 1e2;
    string filename;
    string sampling = "importance_sampling";
    double mean_energy, stddev;
    double beta = 2.82843;
    // double beta = 1;
    VMC my_solver(n_particles, dims, step_sz, alpha, sampling);
    double start = omp_get_wtime();
    double E = my_solver.monte_carlo_sim(mc_samples, therm_samples);
    double end = omp_get_wtime();
    double timeused = end - start;
    std::cout << "timeused = " << timeused << " seconds " << std::endl;
    // exit(1);

    // double start = omp_get_wtime();
    for (int i = 0; i < n; i++){
        alpha = 0.25 + 0.01 * i;
        cout << "alpha = " << alpha << endl;
        // filename = "results/non_interacting/mean_energy_"  + sampling + "_" + to_string(dims) + "_" + to_string(n_particles) + "_" + to_string(i) + ".txt";
        // VMC my_solver(n_particles, dims, step_sz, alpha, sampling);
        filename = "results/interacting/mean_energy_"  + sampling + "_" + to_string(dims) + "_" + to_string(n_particles) + "_" + to_string(i) + ".txt";
        VMC my_solver(n_particles, dims, step_sz, alpha, beta, sampling);
        my_solver.monte_carlo_sim(mc_samples, therm_samples);
        // my_solver.bootstrap(&mean_energy, &stddev, bootstrap_samples);
        // std::cout << "E = " << mean_energy << " ; " << "stddev = " << stddev << std::endl;
        my_solver.write_to_file(filename);
    }
    // double end = omp_get_wtime();
    // double timeused = end-start;
    // cout << "timeused = " << timeused << endl;


    // alpha = 0.5;
    // sampling = "importance_sampling";
    // std::string outfilename = "results/one_body_density_interacting_" + to_string(n_particles) + ".txt";
    // VMC my_solver(n_particles, dims, step_sz, alpha, beta, sampling);
    // double start = omp_get_wtime();
    // my_solver.one_body_density(mc_samples, therm_samples, outfilename);
    // double end = omp_get_wtime();
    // double timeused = end-start;
    // std::cout << "time used = " << timeused << std::endl;

    // double alpha0 = 0.48;
    // int max_iter = 100;
    // double start = omp_get_wtime();
    // optimize(alpha0, max_iter, n_particles);
    // double end = omp_get_wtime();

    return 0;
}


void optimize(double alpha0, int max_iter, int n_particles){
    std::string sampling = "importance_sampling";
    int mc_samples = pow(2,20);
    int therm_samples = 1e2;
    double beta = sqrt(8);
    double step_sz = 0.01;
    int dims = 3;
    double tol = 1e-8;
    int iter = 0;
    double dE = 2*tol;
    double alpha = alpha0;
    double dalpha = 0.01;
    double eta = 0.001;

    VMC my_solver(n_particles, dims, step_sz, alpha, sampling);

    double start = omp_get_wtime();
    double E = my_solver.monte_carlo_sim(mc_samples, therm_samples);
    double end = omp_get_wtime();
    double timeused = end - start;
    std::cout << "timeused = " << timeused << " seconds " << std::endl;

    while (iter < max_iter){
        std::cout << "alpha = " << alpha << std::endl;
        VMC my_solver(n_particles, dims, step_sz, alpha, beta, sampling);
        double E = my_solver.monte_carlo_sim(mc_samples, therm_samples);
        VMC my_solver2(n_particles, dims, step_sz, alpha+dalpha, beta, sampling);
        double E2 = my_solver2.monte_carlo_sim(mc_samples, therm_samples);
        dE = (E2-E)/dalpha;
        alpha -= eta*dE;
        iter++;
    }
    std::cout << "Optimal alpha = " << alpha << std::endl;
}
