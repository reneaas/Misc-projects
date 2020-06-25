#include <iostream>
#include "ising2d.hpp"
#include <cmath>
#include <random>
#include <fstream>
#include <omp.h>
#include <time.h>

using namespace std;

random_device rd;
mt19937_64 gen(rd());


inline int periodic(int coordinate, int dimensions, int step) {
    return (coordinate+dimensions+step) % (dimensions);
}

void Ising2D::InitializeModel(int L, double T)
{
    m_L = L;
    m_Nspins = L*L;
    m_T = T;
    m_beta = 1./T;
    m_spin_matrix = (int*)malloc(m_L*m_L*sizeof(int));

    //Set initial spin state to the ground state of the system
    for (int i = 0; i < m_L; i++){
        for (int j = 0; j < m_L; j++){
            m_spin_matrix[m_L*i + j] = 1;
        }
    }

    //Compute the energy of the system
    m_E = 0;
    m_M = 0;
    for (int i = 0; i < m_L; i++){
        for (int j = 0; j < m_L; j++){
            m_E -= m_spin_matrix[i*m_L + j]*(m_spin_matrix[((i+m_L-1) % m_L)*m_L + j] + m_spin_matrix[i*m_L + (j+m_L-1) % m_L]);
            m_M += m_spin_matrix[i*m_L + j];
        }
    }
}

void Ising2D::Metropolis(int MC, int burn_in)
{
    int id = omp_get_thread_num();
    int cache_line = 42;
    unsigned int seed = id + cache_line;
    int x_flip, y_flip, dE, dM, i;
    double E_sum = 0., M_sum = 0., M_squared = 0., E_squared = 0.;

    int max_int = 10000000;
    double max_inv = 1./max_int;
    int max_index = m_L;
    m_MC = MC;
    m_energies = (double*)malloc(m_MC*sizeof(double));

    //Create lookup table for the boltzmann distribution
    double boltzmann_table[17];
    for (i = -8; i < 9; i++){
        boltzmann_table[i+8] = exp(-m_beta*i);
    }

    //uniform_int_distribution<int> RandomIntegerGenerator(0, m_L-1);        //Sets up the integer distribution for x in [0,n-1]
    //uniform_real_distribution<double> RandomNumberGenerator(0, 1);       //Sets up the uniform distribution for x in (0,1)

    //Burn-in period
    for (i = 0; i < burn_in; i++){
        //x_flip = RandomIntegerGenerator(gen);
        //y_flip = RandomIntegerGenerator(gen);
        x_flip = rand_r(&seed) % max_index;  //Thread safe RNG.
        y_flip = rand_r(&seed) % max_index;  //Thread safe RNB.
        dE = 2*m_spin_matrix[m_L*x_flip + y_flip]*(m_spin_matrix[m_L*( (x_flip + m_L + 1) % m_L ) + y_flip]
                                                    + m_spin_matrix[m_L*( (x_flip + m_L - 1) % m_L ) + y_flip]
                                                    + m_spin_matrix[m_L*x_flip + (y_flip + m_L + 1) % m_L]
                                                    + m_spin_matrix[m_L*x_flip + (y_flip + m_L - 1) % m_L]);

        if ( dE < 0 ){
            m_spin_matrix[x_flip*m_L + y_flip] *= (-1);
            dM = 2*m_spin_matrix[x_flip*m_L + y_flip];
        }
        /*
        //This RNG is not thread safe and will slow down the OpenMP implementation significantly.
        else if( RandomNumberGenerator(gen) < boltzmann_table[dE + 8] ){
            m_spin_matrix[x_flip*m_L + y_flip] *= (-1);
            dM = 2*m_spin_matrix[x_flip*m_L + y_flip];
        }
        */

        //This RNG is thread safe.
        else if ((rand_r(&seed) % max_int)*max_inv < boltzmann_table[dE + 8]){
            m_spin_matrix[x_flip*m_L + y_flip] *= (-1);
            dM = 2*m_spin_matrix[x_flip*m_L + y_flip];
        }
        else{
            dE = 0;
            dM = 0;
        }
        m_E += dE;
        m_M += dM;
    }

    //Once the system reaches its steady state, we move on to perform the actual sampling.
    for (i = 0; i < m_MC; i++){
        //x_flip = RandomIntegerGenerator(gen);
        //y_flip = RandomIntegerGenerator(gen);
        x_flip = rand_r(&seed) % max_index;
        y_flip = rand_r(&seed) % max_index;


        dE = 2*m_spin_matrix[m_L*x_flip + y_flip]*(m_spin_matrix[m_L*( (x_flip + m_L + 1) % m_L ) + y_flip]
                                                    + m_spin_matrix[m_L*( (x_flip + m_L - 1) % m_L ) + y_flip]
                                                    + m_spin_matrix[m_L*x_flip + (y_flip + m_L + 1) % m_L]
                                                    + m_spin_matrix[m_L*x_flip + (y_flip + m_L - 1) % m_L]);

        if ( dE < 0 ){
            m_spin_matrix[x_flip*m_L + y_flip] *= (-1);
            dM = 2*m_spin_matrix[x_flip*m_L + y_flip];
        }
        /*
        else if( RandomNumberGenerator(gen) < boltzmann_table[dE + 8] ){
            m_spin_matrix[x_flip*m_L + y_flip] *= (-1);
            dM = 2*m_spin_matrix[x_flip*m_L + y_flip];
        }
        */


        else if ((rand_r(&seed) % max_int)*max_inv < boltzmann_table[dE + 8]){
            m_spin_matrix[x_flip*m_L + y_flip] *= (-1);
            dM = 2*m_spin_matrix[x_flip*m_L + y_flip];
        }

        else{
            dE = 0;
            dM = 0;
        }

        m_E += dE;
        m_M += dM;

        m_energies[i] = m_E;  //Measure the systems energy.
        E_sum += (double) m_E;
        M_sum += abs(m_M);
        E_squared += m_E*m_E;
        M_squared += m_M*m_M;
    }



    E_sum /= ((double) m_MC);
    E_squared /= m_MC;
    M_sum /= (m_MC*m_Nspins);
    M_squared /= (m_MC*m_Nspins);

    cout << "Average E = " << E_sum/m_Nspins << endl;
    cout << "Average |M| = " << M_sum/m_Nspins << endl;
}


void Ising2D::WriteToFile()
{
    ofstream ofile;
    string outfilename = "energies.txt";
    ofile.open(outfilename, fstream::out);
    for (int i = 0; i < m_MC; i++){
        ofile << m_energies[i] << endl;
    }
    ofile.close();
}

void Ising2D::CleanUp()
{
    free(m_energies);
    free(m_spin_matrix);
}
