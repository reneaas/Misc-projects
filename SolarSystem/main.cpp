#include "solarsystem.hpp"
#include <cmath>
#include <iostream>

using namespace std;


int main(int argc, char const *argv[]) {
    int number_of_objects = 3;
    double total_time = 2.3;
    double stepsize = 0.0001;

    SolarSystem my_system;
    my_system.Initialize(number_of_objects, stepsize);
    //my_system.ReadInitialData();  //Data for the actual solar system, currently only works with 4 objects.
    my_system.InitializeThreeBodyData();  //Data for an artificial 3-bodyproblem with 2 stars (one large and one small) and a small planet.
    cout << "Solving problem" << endl;
    my_system.Solve(total_time);
    cout << "Main program completed..." << endl;
    return 0;
}
