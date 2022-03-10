from libcpp.vector cimport vector # Necessary to use std::vector as inputs and outputs of C++ function.

# Make the .cpp file source code visible to Cython.
# No actions necessary, thus we simply pass it.
# But this line is necessary to make the content visible to Cython.
cdef extern from "solar_storm.cpp":
    pass

# Make the header content visible to Cython.
# Provide declaration of the parts of the code you want to use via Cython.
cdef extern from "solar_storm.hpp":
    cdef vector[double] c_get_B_field(vector[double] r, int num_results)

# Python wrapper around the C/C++ function.
def get_B_field(r, num_results):
    return c_get_B_field(r, num_results)