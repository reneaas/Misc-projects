#include "midpoint.hpp"
#include <omp.h>


double midpoint(double a, double b, int n, double f(double x)) {
    double h = (b - a) * (1. / n);
    double x;
    double res = 0;

    #pragma omp parallel for private(x) reduction(+:res)
    for (int i = 0; i < n-1; i++) {
        x = (i + 0.5)  * h;
        res += f(x);
    }
    res *= h;
    return res;
}