# distutils: language = c++

#Compile with "python3 setup.py build_ext --inplace"

from libc.math cimport exp

cdef extern from "midpoint.cpp":
    pass

cdef extern from "midpoint.hpp":
    cdef double midpoint(double a, double b, int n, double f(double x))
    cdef double mc_integrate(double a, double b, int n, double f(double x))

    
cdef double func(double x):
    return exp(-x)

cpdef integrate(a, b, n):
    res = midpoint(a, b, n, func)
    return res

cpdef integrate_mc(a, b, n):
    res = mc_integrate(a, b, n, func)
    return res
