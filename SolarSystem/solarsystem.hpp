#include <fstream>
#include <cmath>

#ifndef SOLARSYSTEM_HPP
#define SOLARSYSTEM_HPP
using namespace std;

class SolarSystem {
private:
    int m_number_of_objects;
    const int m_dims = 3;
    double *m_masses;
    double *m_vel_old, *m_vel_new, *m_pos_old, *m_pos_new;
    double *m_acc_old, *m_acc_new;
    double m_stepsize, m_stepsize_squared;
    int m_Nsteps;
    double m_total_time;
    double m_total_mass_inv = 0.;
    int m_i, m_j, m_k;
    const double G = 4*M_PI*M_PI;
    ofstream ofile_pos;
    string filename_pos = "computed_positions.txt";

public:
    void Initialize(int number_of_objects, double stepsize);
    void ReadInitialData();
    void InitializeThreeBodyData();
    void Solve(double total_time);
    void AdvancePosition(int j);
    void AdvanceVelocity(int j);
    void SwapPointers();
    void WriteToFile();
    void EulerCromer(int j);
    void ComputeAcceleration(int j);
};

#endif
