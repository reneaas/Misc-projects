from libcpp.vector cimport vector

cdef extern from "solar_system.cpp":
    pass

cdef extern from "solar_system.hpp":
    cdef cppclass SolarSystem:
        SolarSystem() except +
        SolarSystem(vector[double] r0, vector[double] v0, vector[double] m) except +
        vector[double] r_, v_, m_

        vector[double] get_position()
        vector[double] get_velocity()
        vector[double] get_force()
        void step(vector[double] force, double dt)
        vector[double] get_mass()
        int get_num_objects()