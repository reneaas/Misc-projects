#include <cstdlib>
#include <cstdio>
#include <mpi.h>
#include <string>
#include "diffusion_solver.hpp"

using namespace std;

void DiffusionSolver::MPI_Solve(double max_time)
{
  //Set up MPI Universe
  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  printf("my_rank = %d\n", my_rank);
  MPI_Finalize();
}
