#include <cstdlib>
#include <cstdio>
#include <mpi.h>
#include <string>
#include <cmath>
#include "diffusion_solver.hpp"

using namespace std;

void DiffusionSolver::MPI_Solve(double max_time)
{
  //Set up MPI Universe
  int my_rank, comm_sz;
  int i, j;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  //Declare necessary local variables.
  double *v_old, *v_new, ghost_array;
  int M_local, remainder;
  int *n_rows, *sendcount, *displs;

  //Partition data among the processes
  M_local = m_M/comm_sz;
  remainder = m_M % comm_sz;
  n_rows = (int*)malloc(comm_sz*sizeof(int));
  sendcount = (int*)malloc(comm_sz*sizeof(int));
  displs = (int*)malloc(comm_sz*sizeof(int));
  displs[0] = 0;
  sendcount[0] = 0;
  for (int rank = 0; rank < comm_sz-1; rank++){
    n_rows[rank] = (M_local + (remainder > 0));
    sendcount[rank] = n_rows[rank]*m_M;
    displs[rank+1] = displs[rank] + sendcount[rank];
    remainder--;
  }
  n_rows[comm_sz-1] = M_local + (remainder > 0);
  sendcount[comm_sz-1] = n_rows[comm_sz-1]*m_M;

  if (my_rank == 0){
    for (int rank = 0; rank < comm_sz; rank++){
      printf("Rank %d gets %d row(s)\n", rank, n_rows[rank]);
      printf("Rank %d gets %d element(s)\n", rank, sendcount[rank]);
    }
  }


  v_old = (double*)malloc(n_rows[my_rank]*m_M*sizeof(double));
  v_new = (double*)malloc(n_rows[my_rank]*m_M*sizeof(double));

  MPI_Scatterv(m_v_old, sendcount, displs, MPI_DOUBLE, v_old, sendcount[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (my_rank == 0){
    for (i = 0; i < m_M; i++){
      for (j = 0; j < m_M; j++){
        printf("%lf ", m_v_old[i*m_M + j]);
      }
      printf("\n");
    }
    printf("------------------------------------------------------\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 1){
    printf("My rank is %d\n", my_rank);
    for (i = 0; i < n_rows[my_rank]; i++){
      for (j = 0; j < m_M; j++){
        printf("%lf ", v_old[i*m_M + j]);
      }
      printf("\n");
    }
    printf("------------------------------------------------------\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 2){
    printf("My rank is %d\n", my_rank);
    for (i = 0; i < n_rows[my_rank]; i++){
      for (j = 0; j < m_M; j++){
        printf("%lf ", v_old[i*m_M + j]);
      }
      printf("\n");
    }
    printf("------------------------------------------------------\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 3){
    printf("My rank is %d\n", my_rank);
    for (i = 0; i < n_rows[my_rank]; i++){
      for (j = 0; j < m_M; j++){
        printf("%lf ", v_old[i*m_M + j]);
      }
      printf("\n");
    }
    printf("------------------------------------------------------\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  //The actual diffusion solver

  double t = 0, *tmp;
  while (t < max_time){
    for (i = 1; i < M_local-1; i++){
      for (j = 1; j < M_local-1; j++){
        v_new[i*M_local + j] = MPI_ComputeNew_v(i, j, M_local, v_old);
      }
    }
    //Here communication of ghost arrays need to happen before pointer swap occurs.
  }

  MPI_Finalize();
}

double DiffusionSolver::MPI_ComputeNew_v(int i, int j, int M_local, double* v_old)
{
  return (1-4*m_r)*v_old[i*M_local + j]
          + m_r*(v_old[(i+1)*M_local + j] + v_old[(i-1)*M_local + j]
                  + v_old[i*M_local + (j+1)] + v_old[i*M_local + (j-1)] );
}
