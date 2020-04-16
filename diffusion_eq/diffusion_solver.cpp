#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include <omp.h>
#include "diffusion_solver.hpp"
#define pi 3.14159265359

ofstream ofile;

using namespace std;


void DiffusionSolver::Initiate(int N, double r)
{
  //Initialize member variables
  m_N = N;
  m_M = m_N+1;
  m_r = r;
  m_h = 1.0/((double) (N+1)); //Stepsize
  m_dt = r*m_h*m_h;

  //Allocate memory
  m_v_old = (double*)malloc((N+1)*(N+1)*sizeof(double));
  m_v_new = (double*)malloc((N+1)*(N+1)*sizeof(double));
  m_x = (double*)malloc((N+1)*sizeof(double));

  //Set up xy-grid
  for (int i = 0; i < N+1; i++){
    m_x[i] = i*m_h;
  }
}
void DiffusionSolver::SetInitialCondition(double f(double x, double y))
{
  //Set up initial condition
  for (int i = 1; i < m_N; i++){
    for (int j = 1; j < m_N; j++){
      m_v_old[i*m_M + j] = f(m_x[i], m_x[j]);
    }
  }

  free(m_x);

  //Set boundary conditions
  for (int i = 0; i < m_N+1; i++){
    m_v_old[i] = 0.;
    m_v_old[m_M*m_M + i] = 0.;
    m_v_old[i*m_M] = 0.;
    m_v_old[i*m_M + m_M] = 0.;
  }
}

void DiffusionSolver::PrintSolution()
{
  printf("Solution matrix is:\n");
  for (int i = 0; i < m_N+1; i++){
    for (int j = 0; j < m_N+1; j++){
      printf("%lf ", m_v_old[(m_N+1)*i + j]);
    }
    printf("\n");
  }
}

void DiffusionSolver::Solve(double max_time)
{
  double t = 0, timeused;
  double *tmp;

  #if defined(_OPENMP)
  {
    double start, end;
    start = omp_get_wtime();
    while(t < max_time){
      //Do the 2D explicit scheme
      #pragma omp parallel for
      for (int i = 1; i < m_N; i++){
        for (int j = 1; j < m_N; j++){
          m_v_new[i*m_M + j] = ComputeNew_v(i,j);
        }
      }
      //Swap pointers before next time iteration
      tmp = m_v_old;
      m_v_old = m_v_new;
      m_v_new = tmp;
      t += m_dt;
    }
    end = omp_get_wtime();
    timeused = end-start;
    m_final_time = t;
  }
  #else
  {
    clock_t start, end;
    start = clock();
    while(t < max_time){
      //Do the 2D explicit scheme
      for (int i = 1; i < m_N; i++){
        for (int j = 1; j < m_N; j++){
          m_v_new[i*m_M + j] = ComputeNew_v(i,j);
        }
      }
      //Swap pointers before next time iteration
      tmp = m_v_old;
      m_v_old = m_v_new;
      m_v_new = tmp;
      t += m_dt;
    }
    end = clock();
    timeused = (double) (end-start)/CLOCKS_PER_SEC;
    m_final_time = t;
  }
  #endif
  printf("Final timestep = %lf\n", t);
  printf("Time used = %lf\n", timeused);
}


double DiffusionSolver::ComputeNew_v(int i, int j)
{
  return (1-4*m_r)*m_v_old[i*m_M + j]
          + m_r*( m_v_old[(i+1)*m_M + j] + m_v_old[(i-1)*m_M + j]
                  + m_v_old[i*m_M + (j+1)] + m_v_old[i*m_M + (j-1)] );
}

void DiffusionSolver::WriteToFile(string filename)
{
  ofile.open(filename);
  ofile << "time = " << m_final_time << endl;
  for (int i = 0; i < m_N+1; i++){
    for (int j = 0; j < m_N+1; j++){
      ofile << m_v_old[i*m_M + j] << " ";
    }
    ofile << " " << endl;
  }
  ofile.close();

  //Free up memory
  //free(m_v_old);
}
