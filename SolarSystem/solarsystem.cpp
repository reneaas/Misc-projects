#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <armadillo>
#include <string>
#include "solarsystem.hpp"
#include <fstream>

using namespace std;
using namespace arma;


void SolarSystem::Initialize(int number_of_objects, double stepsize)
{
    m_number_of_objects = number_of_objects;
    m_stepsize = stepsize;

    //Initialize matrices.
    m_vel_old = mat(m_number_of_objects, m_dims);
    m_vel_new = mat(m_number_of_objects, m_dims);
    m_pos_old = mat(m_number_of_objects, m_dims);
    m_pos_new = mat(m_number_of_objects, m_dims);
    m_masses = vec(m_number_of_objects);
    m_acceleration = vec(m_dims);
}

void SolarSystem::ReadInitialData()
{
    char* position_file = "positions.txt";
    FILE *fp = fopen(position_file, "r");
    for (int i = 0; i < m_number_of_objects; i++){
        fscanf(fp, "%lf %lf %lf", &m_pos_old(i,0), &m_pos_old(i,1), &m_pos_old(i,2));
    }
    fclose(fp);
    m_pos_old.print("R = ");

    char* velocities_file = "velocities.txt";
    fopen(velocities_file, "r");
    for (int i = 0; i < m_number_of_objects; i++){
        fscanf(fp, "%lf %lf %lf", &m_vel_old(i,0), &m_vel_old(i,1), &m_vel_old(i,2));
    }
    fclose(fp);
    m_vel_old.print("V = ");

    char* masses_file = "masses.txt";
    fopen(masses_file, "r");
    for (int i = 0; i < m_number_of_objects; i++){
        fscanf(fp, "%lf", &m_masses(i));
    }
    fclose(fp);
    m_masses.print("M = ");
}

void SolarSystem::Solve(double total_time)
{
    ofile_pos.open(filename_pos);
    m_total_time = total_time;
    m_Nsteps =  m_total_time/m_stepsize;
    for (int i = 0; i < m_Nsteps; i++){
        cout << "Timestep " << i << " of " << m_Nsteps << "\r";
        for (int j = 0; j < m_number_of_objects; j++){
            ComputeAcceleration(j);
            Advance(j);
        }
        WriteToFile();
    }
    ofile_pos.close();
}

void SolarSystem::ComputeAcceleration(int j)
{
    vec force = vec(m_dims);
    vec total_force = vec(m_dims);
    vec diff_vec = vec(m_dims);
    double rnorm = 0.0;
    force.fill(0.0);
    diff_vec.fill(0.0);
    total_force.fill(0.0);
    m_acceleration.fill(0.0);
    for (int k = 0; k < m_number_of_objects; k++){
        if (k != j){
            for (int l = 0; l < m_dims; l++){
                diff_vec(l) = m_pos_old(j, l) - m_pos_old(k, l);
                force(l) = -diff_vec(l);
            }
            force *= m_masses(k);
            rnorm = norm(diff_vec);
            rnorm = rnorm*rnorm*rnorm;
            for (int l = 0; l < m_dims; l++){
                m_acceleration(l) += force(l)*(1./rnorm);
            }
        }
    }
    m_acceleration *= G;
    //m_acceleration = total_force;
}

void SolarSystem::Advance(int j)
{
    //Advance with Euler-Cromer algorithm
    for (int k = 0; k < m_dims; k++){
        m_vel_new(j, k) = m_vel_old(j, k) + m_acceleration(k)*m_stepsize;
        m_pos_new(j, k) = m_pos_old(j, k) + m_vel_new(j, k)*m_stepsize;

        //Copy new results.
        m_vel_old(j, k) = m_vel_new(j, k);
        m_pos_old(j, k) = m_pos_new(j, k);
    }
}

void SolarSystem::WriteToFile()
{
    for (int i = 0; i < m_number_of_objects; i++){
        for (int j = 0; j < m_dims; j++){
            ofile_pos << m_pos_old(i, j) << " ";
        }
    }
    ofile_pos << " " << endl;
}
