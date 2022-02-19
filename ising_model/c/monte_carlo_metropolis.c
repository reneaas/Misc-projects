#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "ising.h"

void monte_carlo_metropolis(int mc_samples, double temp, int L)
{
    //Create look-up table.
    double boltzmann[17];
    double beta = 1./temp;
    for (int dE = -8; dE < 9; dE+=4){
        boltzmann[dE + 8] = exp(-beta*dE);
    }

    //unsigned int seed = omp_get_thread_num() + 42;

    //Set up ising model
    char *spin_matrix, *idx;
    double E, M;
    initialize_ising(&spin_matrix, &idx, L, &E, &M);
    printf("E = %lf", E);

    //Perform monte carlo simulation using the metropolis sampling rule
    int x, y, dE, max_int = 1e8;
    double max_inv = 1./max_int;
    double E_mean = 0, M_mean = 0, EE_mean = 0, MM_mean = 0;
    for (int n = 0; n < mc_samples; n++){

        //Sample random spin site and compute change in energy
        x = (rand() % L) + 1;
        y = (rand() % L) + 1;
        dE = 2*spin_matrix[idx[x]*L + idx[y]]*(spin_matrix[idx[x+1]*L + idx[y]] +  spin_matrix[idx[x-1]*L + idx[y]] + spin_matrix[idx[x]*L + idx[y+1]] + spin_matrix[idx[x]*L + idx[y-1]]);

        //Consider the transition to the new spin state using metropolis sampling rule
        if (dE < 0){
            //accept spin flip
            E += dE;
            spin_matrix[idx[x]*L + idx[y]] *= (-1);
            M += 2*spin_matrix[idx[x]*L + idx[y]];
        }
        else if ((rand() % max_int)*max_inv < boltzmann[dE + 8]){
            //accept spin flip
            E += 1.*dE;
            spin_matrix[idx[x]*L + idx[y]] *= (-1);
            M += 2*spin_matrix[idx[x]*L + idx[y]];
        }
        //printf("E = %lf\n", E);
        //Add contribution to expectation values
        E_mean += E;
        EE_mean += E*E;
        M_mean += abs(M);
        MM_mean += M*M;
    }
    //Compute mean value of all samples.
    double scaling = 1./(mc_samples*L*L);
    //E_mean *= (1./mc_samples);
    E_mean *= scaling;
    EE_mean *= scaling;
    M_mean *= scaling;
    MM_mean *= scaling;

    printf("<E> = %lf\n", E_mean);
}
