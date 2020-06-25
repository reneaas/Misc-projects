#include <string>
#include <random>
#ifndef ISING2D_HPP
#define ISING2D_HPP

using namespace std;

class Ising2D {
private:
    int m_L, m_MC, m_Ntemps, m_Nspins;
    int *m_spin_matrix;
    int m_E, m_M, m_dE, m_dM;
    double m_beta, m_T;
    string outfilename;
    double m_boltzmann_table[17];
    double *m_energies;



public:
    void InitializeModel(int L, double T);
    void Metropolis(int MC, int burn_in);
    void WriteToFile();
    void CleanUp();
};

#endif
