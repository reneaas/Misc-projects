#include <random>
#include "omp.h"
#include <vector>
#include <cmath>
#include "solar_storm.hpp"

using namespace std;

//Globally set parameters.
double mu0 = 4 * M_PI * 1e-7;
double sigma = 1;
double omega = 1;
double a = 1;


/* 
Computes the Magnetic field of a simple spherical model of Earth
using Monte Carlo integration sampled from a uniform distribution
for each integration variable.
The physical model is assuming a uniformly distributed electric charge
on a sphere rotating with a constant angular velocity.

Args:
    std::vector<double> r:
        Should be a position vector of size 3.
    int num_results:
        Number of results used to compute the integral.
*/
std::vector<double> c_get_B_field(std::vector<double> r, int num_results) {

    double x = r[0], y = r[1], z = r[2]; //Extract position coordinates.
    double B_x = 0., B_y = 0., B_z = 0.; //Initiate field values to zero.
    #pragma omp parallel reduction(+:B_x, B_y, B_z)
    {
        std::mt19937_64 gen(omp_get_thread_num() + 42); // Initialize a generator for each thread.
        std::uniform_real_distribution<double> dist; // U(0, 1) for each thread.
        #pragma omp for
        for (int n = 0; n < num_results; n++) {
            double phi = 2 * M_PI * dist(gen); 
            double theta = M_PI * dist(gen);

            double sin_theta = sin(theta);
            double cos_theta = cos(theta);
            double sin_phi = sin(phi);
            double cos_phi = cos(phi);
            double sin_theta_sq = sin_theta * sin_theta;

            double x_diff = x - a * cos_phi * sin_theta;
            double y_diff = y - a * sin_phi * sin_theta;
            double z_diff = z - a * cos_theta;
            double r_norm = (
                x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
            );
            r_norm = pow(r_norm, 1.5);

            B_x += sin_theta_sq * (
                cos_phi * z_diff
            ) / r_norm;

            B_y += sin_theta_sq * (
                sin_phi * z_diff
            ) / r_norm;

            B_z -= sin_theta_sq * (
                cos_phi * x_diff
                + sin_phi * y_diff
            ) / r_norm;
        }
    }

    double prefactor = 0.5 * mu0 * sigma * omega * M_PI * (1. / num_results);
    std::vector<double> B_field = std::vector<double>(3);
    B_field[0] = B_x * prefactor;
    B_field[1] = B_y * prefactor;
    B_field[2] = B_z * prefactor;
    return B_field;
}