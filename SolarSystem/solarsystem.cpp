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
    m_stepsize_squared = m_stepsize*m_stepsize;

    //Allocate memory for pointers.
    m_vel_old = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));
    m_vel_new = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));
    m_pos_old = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));
    m_pos_new = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));
    m_acc_old = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));
    m_acc_new = (double*)malloc(m_number_of_objects*m_dims*sizeof(double));

    m_masses = (double*)malloc(m_number_of_objects*sizeof(double));
    cout << "Finished allocating" << endl;
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
    fp = fopen(velocities_file, "r");
    for (int i = 0; i < m_number_of_objects; i++){
        fscanf(fp, "%lf %lf %lf", &m_vel_old[i*m_number_of_objects], &m_vel_old[i*m_number_of_objects + 1], &m_vel_old[i*m_number_of_objects + 2]);
    }
    fclose(fp);

    char* masses_file = "masses.txt";
    fp = fopen(masses_file, "r");
    for (int i = 0; i < m_number_of_objects; i++){
        fscanf(fp, "%lf", &m_masses[i]);
    }
    fclose(fp);
    cout << "Finished reading" << endl;
}

void SolarSystem::InitializeThreeBodyData()
{
    double conversion_factor = 31556926./149597871;
    //Planet
    m_pos_old[0] = -1.5;
    m_pos_old[1] = 0.;
    m_pos_old[2] = 0.;
    m_vel_old[0] = 0.;
    m_vel_old[1] = -1*conversion_factor;
    m_vel_old[2] = 0.;
    m_masses[0] = 0.107*5.97e24/1.99e30;

    //Small star
    m_pos_old[m_number_of_objects + 0] = 0.;
    m_pos_old[m_number_of_objects + 1] = 0.;
    m_pos_old[m_number_of_objects + 2] = 0.;
    m_vel_old[m_number_of_objects + 0] = 0.;
    m_vel_old[m_number_of_objects + 1] = 30*conversion_factor;
    m_vel_old[m_number_of_objects + 2] = 0.;
    m_masses[1] = 1.;

    //Large star
    m_pos_old[2*m_number_of_objects + 0] = 3.;
    m_pos_old[2*m_number_of_objects + 1] = 0.;
    m_pos_old[2*m_number_of_objects + 2] = 0.;
    m_vel_old[2*m_number_of_objects + 0] = 0.;
    m_vel_old[2*m_number_of_objects + 1] = -7.5*conversion_factor;
    m_vel_old[2*m_number_of_objects + 2] = 0.;
    m_masses[2] = 4.;
}

void SolarSystem::Solve(double total_time)
{
    ofile_pos.open("computed_positions.txt", fstream::out);
    m_total_time = total_time;
    m_Nsteps =  m_total_time/m_stepsize;
    for (int i = 0; i < m_Nsteps; i++){
        cout << "Timestep " << i << " of " << m_Nsteps << endl;
        for (int j = 0; j < m_number_of_objects; j++){
            AdvancePosition(j);
        }
        for (int j = 0; j < m_number_of_objects; j++){
            AdvanceVelocity(j);
        }
        WriteToFile();
        SwapPointers();
    }
    ofile_pos.close();

    //free(m_masses);
    //free(m_acc_old);
    //free(m_acc_new);

}

void SolarSystem::AdvancePosition(int j)
{
    /*
    This function uses the Velocity-Verlet algorithm.
    */
    double force[m_dims] = {0., 0., 0.};
    double diff_vec[m_dims] = {0., 0., 0.};
    double acc[m_dims] = {0., 0., 0.};
    double mass, rnorm, Grnorm_inv;


    //Compute acceleration on object j
    for (int k = 0; k < m_number_of_objects; k++){
        if (k != j){
            rnorm = 0.;
            mass = m_masses[k];
            for (int l = 0; l < m_dims; l++){
                diff_vec[l] = m_pos_old[j*m_number_of_objects + l] - m_pos_old[k*m_number_of_objects + l];
                force[l] = -mass*diff_vec[l];
                rnorm += diff_vec[l]*diff_vec[l];
            }
            rnorm = pow(rnorm, 1.5);
            Grnorm_inv = G*(1./rnorm);

            for (int l = 0; l < m_dims; l++){
                acc[l] += force[l]*Grnorm_inv;
            }
        }
    }


    for (int l = 0; l < m_dims; l++) m_acc_old[j*m_number_of_objects + l] = acc[l];

    //Update position of the object
    for (int k = 0; k < m_dims; k++){
        m_pos_new[j*m_number_of_objects + k] = m_pos_old[j*m_number_of_objects + k]
                                                + m_vel_old[j*m_number_of_objects + k]*m_stepsize
                                                + 0.5*m_acc_old[j*m_number_of_objects + k]*m_stepsize_squared;
    }
}


void SolarSystem::AdvanceVelocity(int j)
{
    double force[m_dims] = {0., 0., 0.};
    double diff_vec[m_dims] = {0., 0., 0.};
    double acc[m_dims] = {0., 0., 0.};
    double mass, rnorm, Grnorm_inv;
    //Compute acceleration using the new positions:
    for (int k = 0; k < m_number_of_objects; k++){
        mass = m_masses[k];
        if (k != j){
            rnorm = 0.;
            mass = m_masses[k];
            for (int l = 0; l < m_dims; l++){
                diff_vec[l] = m_pos_new[j*m_number_of_objects + l] - m_pos_new[k*m_number_of_objects + l];
                force[l] = -mass*diff_vec[l];
                rnorm += diff_vec[l]*diff_vec[l];
            }
            rnorm = pow(rnorm, 1.5);
            Grnorm_inv = G*(1./rnorm);

            for (int l = 0; l < m_dims; l++){
                acc[l] += force[l]*Grnorm_inv;
            }
        }
    }

    for (int l = 0; l < m_dims; l++) m_acc_new[j*m_number_of_objects + l] = acc[l];

    //Compute new velocities
    for (int k = 0; k < m_number_of_objects; k++){
        m_vel_new[j*m_number_of_objects + k] = m_vel_old[j*m_number_of_objects + k]
                                                + 0.5*(m_acc_old[j*m_number_of_objects + k]
                                                + m_acc_new[j*m_number_of_objects + k])*m_stepsize_squared;
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
            ofile_pos << m_pos_new[i*m_number_of_objects + j] << " ";
        }
    }
    ofile_pos << " " << endl;
}
