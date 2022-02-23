# distutils: language = c++

from libcpp.vector cimport vector
from ising cimport Ising2D
import numpy as np
cimport numpy as np


cdef class PyIsing2D:
    cdef Ising2D ising

    def __cinit__(self, vector[vector[double]] spin_matrix, double T):
        self.ising = Ising2D(spin_matrix, T)
    
    def get_energy(self):
        return self.ising.get_energy()

    def get_magnetization(self):
        return self.ising.get_magnetization()

    def __str__(self):
        return str(np.asarray(self.ising.get_spin_matrix()))


    
