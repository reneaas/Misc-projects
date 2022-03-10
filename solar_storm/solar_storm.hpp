#ifndef SOLAR_STORM_HPP
#define SOLAR_STORM_HPP

#include <random>
#include "omp.h"
#include <vector>

std::vector<double> c_get_B_field(std::vector<double> r, int num_results);

#endif