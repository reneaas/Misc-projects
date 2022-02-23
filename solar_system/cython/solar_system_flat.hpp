#ifndef SOLAR_SYSTEM_FLAT_HPP
#define SOLAR_SYSTEM_FLAT_HPP

#define PI 3.14159265359

#include <vector>
#include <cmath>

class SolarSystemFlat {
private:
    /* data */
    std::vector<double> r_, v_; //Position and velocity.
    std::vector<double> m_; // Mass
    double G_;
    int num_objects_, dims_;
    std::vector<double> force_;


public:
    SolarSystemFlat ();
    SolarSystemFlat (
        std::vector<double> r0,
        std::vector<double> v0,
        std::vector<double> m
    );
    virtual ~SolarSystemFlat (){};

    std::vector<double> get_force();
    void step(std::vector<double> force, double dt);
    std::vector<double> get_position();
    std::vector<double> get_velocity();
    std::vector<double> get_mass();
    int get_num_objects();


};

#endif
