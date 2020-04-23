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
  int *n_elems, *sendcount, *displs;

  //Partition data among the processes
  M_local = m_M*m_M/comm_sz;
  remainder = m_M*m_M % comm_sz;
  n_elems = (int*)malloc(comm_sz*sizeof(int));
  for (int rank = 0; rank < comm_sz; rank++){
    n_elems[rank] = (M_local + (remainder > 0));
    remainder--;
  }

  if (my_rank == 0){
    for (int rank = 0; rank < comm_sz; rank++){
      printf("Rank %d gets %d element(s)\n", rank, n_elems[rank]);
    }
  }


  v_old = (double*)malloc(n_elems[my_rank]*sizeof(double));
  v_new = (double*)malloc(n_elems[my_rank]*sizeof(double));

  int row_elems = sqrt(n_elems[my_rank]);
  int col_elems = row_elems;

  if (my_rank == 0){
    int cumulative_rows;
    for (int rank = 1; rank < comm_sz; rank++){
      row_elems = sqrt(n_elems[rank]);
      col_elems = row_elems;
      if (rank % 2 != 0){
        for (i = 0; i < row_elems; i++){
          for (j = 0; j < col_elems; j++){
            MPI_Send(&m_v_old[(i+cumulative_rows)*col_elems + (col_elems+j)], 1, MPI_DOUBLE, rank, rank, MPI_COMM_WORLD);
          }
        }
      }
      if (rank % 2 == 0){
        for (i = 0; i < row_elems; i++){
          for (j = 0; j < col_elems; j++){
            MPI_Send(&m_v_old[(i+cumulative_rows)*col_elems + (col_elems+j)], 1, MPI_DOUBLE, rank, rank, MPI_COMM_WORLD);
          }
        }
        cumulative_rows += row_elems;
      }
    }
  }
  else{
    for (i = 0; i < row_elems; i++){
      for (j = 0; j < col_elems; j++){
        MPI_Recv(&v_old[i*col_elems + j], 1, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }

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
    for (i = 0; i < row_elems; i++){
      for (j = 0; j < col_elems; j++){
        printf("%lf ", v_old[i*col_elems + j]);
      }
      printf("\n");
    }
    printf("------------------------------------------------------\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 2){
    printf("My rank is %d\n", my_rank);
    for (i = 0; i < row_elems; i++){
      for (j = 0; j < col_elems; j++){
        printf("%lf ", v_old[i*col_elems + j]);
      }
      printf("\n");
    }
    printf("------------------------------------------------------\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 3){
    printf("My rank is %d\n", my_rank);
    for (i = 0; i < row_elems; i++){
      for (j = 0; j < col_elems; j++){
        printf("%lf ", v_old[i*col_elems + j]);
      }
      printf("\n");
    }
    printf("------------------------------------------------------\n");
  }





  //MPI_Scatterv(m_v_old, n_elems, displs, MPI_DOUBLE, v_old, n_elems[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD); //Scatters the data among the processes.


  MPI_Finalize();
}
