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
    double m_stepsize;
    int m_Nsteps;
    double m_total_time;
    int m_i, m_j, m_k;
    const double G = 4*M_PI*M_PI;
    ofstream ofile_pos, ofile_vel;
    string filename_pos = "computed_positions.txt";
    string filename_vel = "computed_velocities.txt";

public:
    void Initialize(int number_of_objects, double stepsize);
    void ReadInitialData();
    void Solve(double total_time);
    void Advance(int j);
    void SwapPointers();
    void WriteToFile();
};

#endif
