#include "ising.h"
#include <stdio.h>
#include <omp.h>


int main(int argc, char const *argv[]) {
    int L = 2;
    double temp = 1.0;
    int mc_samples = 1e8;
    double T_start = 2.0, T_end = 2.4;
    int n_temps = 10;
    double h = (T_end-T_start)/(n_temps-1);

    double start = omp_get_wtime();
    for (int L = 40; L <= 100; L+=20){
        printf("L = %d\n", L);
        #pragma omp parallel for
        for (int i = 0; i < n_temps; i++){
            double T = T_start + i*h;
            omp_monte_carlo_metropolis(mc_samples, T, L);
        }
    }
    double end = omp_get_wtime();
    double timeused = end-start;
    printf("Timeused  = %lf seconds\n", timeused);


    return 0;
}
