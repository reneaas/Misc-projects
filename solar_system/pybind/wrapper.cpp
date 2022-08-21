
#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // automatic conversion between vector/valarray and numpy array
#include <pybind11/functional.h> // automatic conversion between functional and python function
#include "solar_system.hpp"

// SolarSystem (
//         std::vector<double> r0,
//         std::vector<double> v0,
//         std::vector<double> m
//     );

// std::vector<double> get_force();
// void step(std::vector<double> force, double dt);
// std::vector<double> get_position();
// std::vector<double> get_velocity();
// std::vector<double> get_mass();
// int get_num_objects();


namespace py = pybind11;

PYBIND11_MODULE(SolarSystem, m) {
    py::class_<SolarSystem>(m, "Solar System", py::dynamic_attr())
        .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>> );
        .def("get_force", &SolarSystem::get_force);
        .def("step", &SolarSystem::step);
        .def("get_position", &SolarSystem::get_position);
        .def("get_velocity", &SolarSystem::get_velocity);
        .def("get_mass"), &SolarSystem::get_mass);
        .def("get_num_objects", &SolarSystem::get_num_objects);
}

