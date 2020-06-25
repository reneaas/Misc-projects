#include <iostream>
#include "ising2d.hpp"
#include <mpi.h>
#include <time.h>

int main(int argc, char const *argv[]) {

    int L = 20;
    double T;
    int burn_in = 1000;
    int MC = 100000000;
    double start, end, timeused;
    double start_temp = 2.0, end_temp = 2.4;
    int num_temps = 20;
    double temp_step = (end_temp-start_temp)/((double) num_temps);

    int my_rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int *temp_distribution = (int*)malloc(comm_sz*sizeof(int));

    int local_num_temps = num_temps/comm_sz;
    int remainder = num_temps % comm_sz;

    for (int i = 0; i < comm_sz; i++){
        temp_distribution[i] = local_num_temps + (remainder > 0);
        remainder--;
    }
    int my_interval_length = temp_distribution[my_rank];
    free(temp_distribution);

    start_temp = start_temp + my_interval_length*my_rank*temp_step;
    end_temp = start_temp +  my_interval_length*temp_step;
    cout << "start temp = " << start_temp << " for rank" << my_rank << endl;
    cout << "end temp = " << end_temp << " for rank" << my_rank << endl;
    cout << "rank " << my_rank << " got " << my_interval_length << " temperatures" << endl;

    start = MPI_Wtime();
    for (int i = 0; i < my_interval_length; i++){
        T = start_temp + i*temp_step;
        Ising2D my_model;
        my_model.InitializeModel(L, T);
        my_model.Metropolis(MC, burn_in);
        my_model.CleanUp();
        //my_model.WriteToFile();
    }
    end = MPI_Wtime();
    timeused = end-start;

    double max_time;
    MPI_Reduce(&timeused, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) cout << "timeused = " << max_time << " seconds" << endl;

    MPI_Finalize();
    return 0;
}
