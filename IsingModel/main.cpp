#include <iostream>
#include "ising2d.hpp"
#include <omp.h>
#include <time.h>

int main(int argc, char const *argv[]) {
    int L = 20;
    double T = 1.;
    int burn_in = 1000;
    int MC = 100000000;
    double timeused;

    #ifdef _OPENMP
    {
        double start, end;
        int num_temps = 20;
        double start_temp = 2.0;
        double end_temp = 2.4;
        double temp_step = (end_temp-start_temp)/num_temps;

        start = omp_get_wtime();
        #pragma omp parallel for private(T)
        for (int i = 0; i < num_temps; i++){
            T = start_temp + i*temp_step;
            //cout << "Simulating for T = " << T << endl;
            Ising2D my_model;
            my_model.InitializeModel(L, T);
            my_model.Metropolis(MC, burn_in);
            my_model.CleanUp();
            //my_model.WriteToFile();
        }
        //my_model.WriteToFile();
        end = omp_get_wtime();
        timeused = end-start;
    }
    #else
    {
        clock_t start, end;
        start = clock();
        Ising2D my_model;
        my_model.InitializeModel(L, T);
        my_model.Metropolis(MC, burn_in);
        end = clock();
        timeused = (double) (end-start)/CLOCKS_PER_SEC;
    }
    #endif

    cout << "Timeused = " << timeused << " seconds" << endl;
    return 0;
}
