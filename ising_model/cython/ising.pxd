# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "ising2d.cpp":
    pass

cdef extern from "ising2d.hpp":
    cdef cppclass Ising2D:
        Ising2D() except +
        Ising2D(vector[vector[double]] spin_matrix, double T) except +

        double get_energy()
        double get_magnetization()
        vector[vector[double]] get_spin_matrix()

