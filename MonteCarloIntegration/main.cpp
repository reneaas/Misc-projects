#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <time.h>
#include "MonteCarloIntegrator.hpp"
#include <omp.h>

using namespace std;

double f(double);

int main(int argc, char const *argv[]) {
  int MC_samples = 1024;
  double a = 0;
  double b = 100;
  MonteCarloIntegrator Integrator;
  Integrator.Initiate(a, b, MC_samples);
  Integrator.UniformIntegration(f);
  return 0;
}

double f(double x)
{
  return exp(-x);
}
