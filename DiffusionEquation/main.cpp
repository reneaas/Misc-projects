#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include "diffusion_solver.hpp"
#define pi 3.14159265359

using namespace std;

double f(double x, double y);


/*
This main program provides a test of the
class DiffusionSolver to showcase its use.
This version can be run with or without
parallelization with OpenMP.

To run serial code, type in a linux terminal:

make serial

To run code parallelized with OpenMP, type in a linux terminal:

make omp


*/
int main(int argc, char const *argv[]) {
  int N = 100;
  double r = 0.25;
  double max_time = 1;
  string filename = "output.txt";

  DiffusionSolver my_solver;
  my_solver.Initiate(N, r);
  my_solver.SetInitialCondition(f);
  //my_solver.PrintSolution();
  my_solver.Solve(max_time);
  //my_solver.MPI_Solve(max_time);
  //my_solver.PrintSolution();
  my_solver.WriteToFile(filename);

  return 0;
}

/*
The chosen initial condition of the 2D-diffusion equation.
This is chosen for simplicity since the initial condition
conincides with a single term in the general Fourier series
of the analytical solution such that
v(x,y,t) = f(x,y)*exp(-2*pi^2*t).
*/
double f(double x, double y)
{
  return sin(pi*x)*sin(pi*y);
}
