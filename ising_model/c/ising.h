#ifndef ISING_H
#define ISING_H

void initialize_ising(char **spin_matrix, char **idx, int L, double *E, double *M);
void monte_carlo_metropolis(int mc_samples, double temp, int L);
void omp_monte_carlo_metropolis(int mc_samples, double temp, int L);

#endif
