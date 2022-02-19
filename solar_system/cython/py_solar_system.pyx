# distutils: language = c++

from libcpp.vector cimport vector
from py_solar_system cimport SolarSystem

cdef class PySolarSystem:
    cdef SolarSystem solar_system

    def __cinit__(self, vector[vector[double]] r0, vector[vector[double]] v0, vector[double] m):
        self.solar_system = SolarSystem(r0, v0, m)
    
    def get_position(self):
        return self.solar_system.get_position()

    def get_velocity(self):
        return self.solar_system.get_velocity()

    def get_force(self):
        return self.solar_system.get_force()

    def step(self, vector[vector[double]] force, double dt):
        self.solar_system.step(force, dt)

    @property
    def num_objects(self):
        return self.solar_system.get_num_objects()

    @property
    def m(self):
        return self.solar_system.get_mass()
