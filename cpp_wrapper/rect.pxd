cdef extern from "rectangle.cpp":
    pass

cdef extern from "rectangle.hpp" namespace "shapes":
    cdef cppclass Rectangle:
        Rectangle() except +
        Rectangle(int, int, int, int) except +
        int x0, x1, y0, y1
        int getArea()
        void getSize(int* width, int* heigth)
        void move(int, int)