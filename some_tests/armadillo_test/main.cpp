#include <cstdlib>
#include <cstdio>
#include <armadillo>
#include <time.h>
#include <omp.h>

using namespace std;
using namespace arma;

int main(){
    double start, end;
    double timeused;
    int N = 10000;
    mat A = mat(N,N);
    A.randn();
    printf("Finding the inverse now...\n");
    start = omp_get_wtime();
    mat A_inv = mat(N,N);
    A_inv = inv(A);
    end = omp_get_wtime();
    timeused = end - start; 
    printf("Time used = %lf\n", timeused);
    printf("Finished.\n");
}
