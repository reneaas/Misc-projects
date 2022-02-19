#include "midpoint.hpp"
#include <omp.h>
#include <random>


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

double mc_integrate(double a, double b, int n, double f(double x)) {
    std::mt19937_64 gen;
    std::uniform_real_distribution<double> dist(a, b);

    double x;
    double res = 0;
    #pragma omp parallel for private(x) reduction(+:res)
    for (int i = 0; i < n; i++) {
        x = (b - a) * dist(gen);
        res += f(x);
    }
    res *= (1. / n);
    res *= (b - a);
    return res;
}