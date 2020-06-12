#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include "solarsystem.hpp"
#include <fstream>
#include <iostream>

using namespace std;

void SolarSystem::Initialize(int number_of_objects, double stepsize)
{
    m_number_of_objects = number_of_objects;
    m_stepsize = stepsize;

    //Allocate memory for pointers.
    m_vel_old = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));
    m_vel_new = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));
    m_pos_old = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));
    m_pos_new = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));

    m_masses = (double*)malloc(m_number_of_objects*sizeof(double));
}

void SolarSystem::ReadInitialData()
{
    char* position_file = "positions.txt";
    FILE *fp = fopen(position_file, "r");
    for (int i = 0; i < m_number_of_objects; i++){
        fscanf(fp, "%lf %lf %lf", &m_pos_old[i*m_number_of_objects], &m_pos_old[i*m_number_of_objects + 1], &m_pos_old[i*m_number_of_objects + 2]);
    }
    fclose(fp);

    char* velocities_file = "velocities.txt";
    fopen(velocities_file, "r");
    for (int i = 0; i < m_number_of_objects; i++){
        fscanf(fp, "%lf %lf %lf", &m_vel_old[i*m_number_of_objects], &m_vel_old[i*m_number_of_objects + 1], &m_vel_old[i*m_number_of_objects + 2]);
    }
    fclose(fp);

    char* masses_file = "masses.txt";
    fopen(masses_file, "r");
    for (int i = 0; i < m_number_of_objects; i++){
        fscanf(fp, "%lf", &m_masses[i]);
    }
    fclose(fp);
}

void SolarSystem::Solve(double total_time)
{
    ofstream ofile_pos;
    ofile_pos.open(filename_pos);
    m_total_time = total_time;
    m_Nsteps =  m_total_time/m_stepsize;
    for (int i = 0; i < m_Nsteps; i++){
        cout << "Timestep " << i << " of " << m_Nsteps << endl;
        for (int j = 0; j < m_number_of_objects; j++){
            Advance(j);
        }
        WriteToFile();
        SwapPointers();
    }
    ofile_pos.close();
}

void SolarSystem::Advance(int j)
{
    double force[m_dims] = {0., 0., 0.};
    double diff_vec[m_dims] = {0., 0., 0.};
    double acceleration[m_dims] = {0., 0., 0.};
    double mass, rnorm, Grnorm_inv;


    //Compute acceleration on object j
    for (int k = 0; k < m_number_of_objects; k++){
        rnorm = 0.;
        //cout << "Compute force from object " << k << "on object" << j << endl;
        mass = m_masses[k];
        if (k != j){
            for (int l = 0; l < m_dims; l++){
                diff_vec[l] = m_pos_old[j*m_number_of_objects + l] - m_pos_old[k*m_number_of_objects + l];
                force[l] = -mass*diff_vec[l];
                rnorm += diff_vec[l]*diff_vec[l];
            }
            rnorm = pow(rnorm, 1.5);
            Grnorm_inv = G*(1./rnorm);

            for (int l = 0; l < m_dims; l++){
                acceleration[l] += force[l]*Grnorm_inv;
            }
        }
    }

    //Advance object j with Euler-Cromer algorithm
    for (int k = 0; k < m_dims; k++){
        m_vel_new[j*m_number_of_objects + k] = m_vel_old[j*m_number_of_objects + k] + acceleration[k]*m_stepsize;
        m_pos_new[j*m_number_of_objects + k] = m_pos_old[j*m_number_of_objects + k] + m_vel_new[j*m_number_of_objects + k]*m_stepsize;
    }
}

void SolarSystem::SwapPointers()
{
    double *tmp_pos, *tmp_vel;

    //Swap position pointers.
    tmp_pos = m_pos_old;
    m_pos_old = m_pos_new;
    m_pos_new = tmp_pos;

    //Swap velocity pointers
    tmp_vel = m_vel_old;
    m_vel_old = m_vel_new;
    m_vel_new = tmp_vel;
}


void SolarSystem::WriteToFile()
{
    for (int i = 0; i < m_number_of_objects; i++){
        for (int j = 0; j < m_dims; j++){
            ofile_pos << m_pos_old[i*m_number_of_objects + j] << " ";
        }
    }
    ofile_pos << " " << endl;
}
