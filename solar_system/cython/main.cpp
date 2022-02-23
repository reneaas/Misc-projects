#include "solar_system.hpp"
#include <iostream>
#include <armadillo>
#include <vector>
#include "time.h"

using namespace std;


int main() {

    arma::mat init_pos, init_vel, mass;
    init_pos.load("../data/pos.bin");
    init_vel.load("../data/vel.bin");
    mass.load("../data/mass.bin");

    std::vector< std::vector<double> > r0, v0;
    std::vector<double> m;

    r0.resize(init_pos.n_cols);
    v0.resize(init_pos.n_cols);
    m.resize(init_pos.n_cols);

    for (int i = 0; i < init_pos.n_cols; i++) {
        r0[i].push_back(init_pos(0, i));
        r0[i].push_back(init_pos(1, i));
        r0[i].push_back(init_pos(2, i));

        v0[i].push_back(init_vel(0, i));
        v0[i].push_back(init_vel(1, i));
        v0[i].push_back(init_vel(2, i));

        m[i] = mass(i);
    }


    SolarSystem solar_system = SolarSystem(r0, v0, m);

    int num_timesteps = (int) 1e6;
    double dt = 0.001;

    clock_t start = clock();
    vector<vector<vector<double> > > R = solar_system.compute_evolution(num_timesteps, dt);
    clock_t end = clock();
    double timeused = (end - start) * (1. / CLOCKS_PER_SEC);
    cout << "Timeused = " << timeused << endl;

}