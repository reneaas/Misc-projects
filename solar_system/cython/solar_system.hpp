#ifndef SOLAR_SYSTEM_HPP
#define SOLAR_SYSTEM_HPP

#define pi 3.14159265359

#include <vector>

class SolarSystem {
private:
    /* data */
    std::vector< std::vector<double> > r_, v_; //Position and velocity.
    std::vector<double> m_; // Mass

    double G_;


public:
    SolarSystem ();
    SolarSystem (
        std::vector< std::vector<double> > r0,
        std::vector< std::vector<double> > v0,
        std::vector<double> m
    );
    virtual ~SolarSystem (){};

    std::vector< std::vector<double> > get_force();
    void step(std::vector< std::vector<double> > force, double dt);

    std::vector< std::vector<double> > get_position();
    std::vector< std::vector<double> > get_velocity();
    std::vector<double> get_mass();
    int get_num_objects();


};

#endif
