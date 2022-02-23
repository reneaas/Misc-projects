# distutils: language = c++

from libcpp.vector cimport vector
from py_solar_system cimport SolarSystem, SolarSystemFlat
import numpy as np
cimport numpy as np

cdef class PySolarSystem:
    cdef SolarSystem solar_system #Holds the C++ object.

    def __cinit__(self, vector[vector[double]] r0, vector[vector[double]] v0, vector[double] m):
        self.solar_system = SolarSystem(r0, v0, m)
    
    def get_position(self):
        """Returns the current positions of the objects in the solar system."""
        return self.solar_system.get_position()

    def get_velocity(self):
        """Returns the current velocities of the objects in the solar system."""
        return self.solar_system.get_velocity()

    def get_force(self):
        """Returns the current force on each object in the solar system."""
        return self.solar_system.get_force()

    def step(self, vector[vector[double]] force, double dt):
        """Computes one step using Euler-Cromer.
        
        Args:
            force:
                2D numpy array of shape [num_particles, num_dims]
            dt:
                Step size to use with the integrator.
        """
        self.solar_system.step(force, dt)
    
    def compute_evolution(self, int num_timesteps, double dt):
        return self.solar_system.compute_evolution(num_timesteps, dt)

    @property
    def num_objects(self):
        return self.solar_system.get_num_objects()

    @property
    def m(self):
        return self.solar_system.get_mass()


cdef class PySolarSystemFlat:
    cdef SolarSystemFlat solar_system #Holds the C++ object.

    def __cinit__(self, vector[double] r0, vector[double] v0, vector[double] m):
        self.solar_system = SolarSystemFlat(r0, v0, m)
    
    def get_position(self):
        """Returns the current positions of the objects in the solar system."""
        return self.solar_system.get_position()

    def get_velocity(self):
        """Returns the current velocities of the objects in the solar system."""
        return self.solar_system.get_velocity()

    def get_force(self):
        """Returns the current force on each object in the solar system."""
        return self.solar_system.get_force()

    def step(self, vector[double] force, double dt):
        """Computes one step using Euler-Cromer.
        
        Args:
            force:
                2D numpy array of shape [num_particles, num_dims]
            dt:
                Step size to use with the integrator.
        """
        self.solar_system.step(force, dt)

    @property
    def num_objects(self):
        return self.solar_system.get_num_objects()

    @property
    def m(self):
        return self.solar_system.get_mass()