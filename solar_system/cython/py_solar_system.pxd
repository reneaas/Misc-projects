# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "solar_system.cpp":
    pass

cdef extern from "solar_system.hpp":
    cdef cppclass SolarSystem:
        SolarSystem() except +
        SolarSystem(vector[vector[double]] r0, vector[vector[double]] v0, vector[double] m) except +
        vector[vector[double]] r_, v_
        vector[double] m_

        vector[vector[double]] get_position()
        vector[vector[double]] get_velocity()
        vector[vector[double]] get_force()
        void step(vector[vector[double]] force, double dt)
        vector[double] get_mass()
        int get_num_objects()
