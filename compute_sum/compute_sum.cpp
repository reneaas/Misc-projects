#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char const *argv[]) {
    double s = 0;
    int n = 100;
    #ifdef _OPENMP
    {
        #pragma omp parallel
        {
            double local_s = 0;
            #pragma omp for
            for (int i = 0; i <= n; i++){
                local_s += i;
            }

            #pragma omp atomic
                s += local_s;
        }
    }
    #else
    {
        for (int i = 0; i <= n; i++){
            s += i;
        }
    }
    #endif

    cout << "sum of 100 first integers = " << s << endl;
    return 0;
}
