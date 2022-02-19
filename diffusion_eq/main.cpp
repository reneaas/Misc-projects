#include "diffusion2d.hpp"
#include "diffusion_base.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>
#define pi 3.14159265359

double initial_fn(double x, double y);

int main(int argc, char const *argv[]) {
    int n = 250;
    double r = 0.25;
    double tot_time = 1;

    Diffusion2D my_solver(n, r);
    my_solver.init_cond(initial_fn);
    double start = omp_get_wtime();
    my_solver.solve(tot_time);
    double end = omp_get_wtime();
    double timeused = end-start;
    std::cout << "timeused = " << timeused << " seconds " << std::endl;


    return 0;
}

/*
The chosen initial condition of the 2D-diffusion equation.
This is chosen for simplicity since the initial condition
coincides with a single term in the general Fourier series
of the analytical solution such that
v(x,y,t) = f(x,y)*exp(-2*pi^2*t).
*/
double initial_fn(double x, double y)
{
  return sin(pi*x)*sin(pi*y);
}
