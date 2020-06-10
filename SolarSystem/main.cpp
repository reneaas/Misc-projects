#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <armadillo>
#include <string>
#include "solarsystem.hpp"

using namespace arma;
using namespace std;


int main(int argc, char const *argv[]) {
    int number_of_objects = 10;
    double total_time = 1;
    double stepsize = 0.0001;

    SolarSystem my_system;
    my_system.Initialize(number_of_objects, stepsize);
    my_system.ReadInitialData();
    my_system.Solve(total_time);
    return 0;
}
