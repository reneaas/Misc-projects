#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <time.h>
#include "MonteCarloIntegrator.hpp"
#include <omp.h>

using namespace std;

void MonteCarloIntegrator::Initiate(double a, double b, int N_MCsamples)
{
  m_a = a;
  m_b = b;
  m_MCsamples = N_MCsamples;
}

void MonteCarloIntegrator::UniformIntegration(double f(double x))
{
  int i, rand_max;
  double u, x, normalizing_factor, Integral = 0.;
  rand_max = 100000;
  normalizing_factor = 1./((double) rand_max);

  #ifdef _OPENMP
  {
    rand_max = 100000;
    normalizing_factor = 1./((double) rand_max);
    //unsigned int seed = time(NULL);

    #pragma omp parallel private(i, u, x)
    {
      double start = omp_get_wtime();
      int id = omp_get_thread_num();
      int cache_line = 42;
      unsigned int seed = id + cache_line;
      #pragma omp for reduction(+:Integral)
      for (i = 0; i < m_MCsamples; i++){
        u = (rand() % rand_max)*normalizing_factor;   //Not thread-safe RNG.
        //u = (rand_r(&seed) % rand_max)*normalizing_factor; //Thread-safe RNG.
        x = (m_b - m_a)*u + m_a;
        Integral += f(x);
      }
      double end = omp_get_wtime();
      double timeused = end-start;
      printf("timeused = %lf\n", timeused);
    }
    Integral = Integral / ((double) m_MCsamples);
    printf("Integral = %lf\n", Integral);

  }
  #else
  {
    srand(time(0));
    for (i = 0; i < m_MCsamples; i++){
      u = (rand() % rand_max)*normalizing_factor;
      printf("u = %lf\n", u);
      x = (m_b - m_a)*u + m_a;
      Integral += f(x);
    }
    Integral /= ((double) m_MCsamples);
    printf("Integral = %lf\n", Integral);
  }
  #endif
}
