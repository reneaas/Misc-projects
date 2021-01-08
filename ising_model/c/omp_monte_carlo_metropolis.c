#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "ising.h"
#include <omp.h>


void omp_monte_carlo_metropolis(int mc_samples, double temp, int L)
{
    unsigned int seed = omp_get_thread_num() + 42; //Give the thread a seed.

    //Create look-up table.
    double boltzmann[17];
    double beta = 1./temp;
    for (int dE = -8; dE < 9; dE+=4){
        boltzmann[dE + 8] = exp(-beta*dE);
    }


    //Set up ising model
    char *spin_matrix, *idx;
    double E, M;
    initialize_ising(&spin_matrix, &idx, L, &E, &M);

    //Perform monte carlo simulation using the metropolis sampling rule
    int x, y, dE, max_int = 1e8;
    double max_inv = 1./max_int;
    double E_mean = 0, M_mean = 0, EE_mean = 0, MM_mean = 0;
    for (int n = 0; n < mc_samples; n++){

        //Sample random spin site and compute change in energy
        x = (rand_r(&seed) % L) + 1;
        y = (rand_r(&seed) % L) + 1;
        dE = 2*spin_matrix[idx[x]*L + idx[y]]*(spin_matrix[idx[x+1]*L + idx[y]] +  spin_matrix[idx[x-1]*L + idx[y]] + spin_matrix[idx[x]*L + idx[y+1]] + spin_matrix[idx[x]*L + idx[y-1]]);

        //Consider the transition to the new spin state using metropolis sampling rule
        if (dE < 0){
            //accept spin flip
            E += dE;
            spin_matrix[idx[x]*L + idx[y]] *= (-1);
            M += 2*spin_matrix[idx[x]*L + idx[y]];
        }
        else if ((rand_r(&seed) % max_int)*max_inv < boltzmann[dE + 8]){
            //accept spin flip
            E += dE;
            spin_matrix[idx[x]*L + idx[y]] *= (-1);
            M += 2*spin_matrix[idx[x]*L + idx[y]];
        }
        //Add contribution to expectation values
        E_mean += E;
        EE_mean += E*E;
        M_mean += abs(M);
        MM_mean += M*M;
    }
    //Compute mean value of all samples.
    double mc_inv = 1./mc_samples;
    double n_spins_inv = 1./(L*L);
    double norm = mc_inv*n_spins_inv;
    E_mean *= norm;
    EE_mean *= norm;
    M_mean *= norm;
    MM_mean *= norm;

    //printf("<E> = %lf\n", E_mean);


    free(spin_matrix);
    free(idx);
}
