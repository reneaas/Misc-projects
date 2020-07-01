from numba import jit, prange
import numpy as np

a = np.linspace(0, 13, 100000001)
b = np.linspace(0, 13, 100000001)

@jit(parallel=True)
def f(a,b):
    s = 0
    for i in prange(len(a)):
        s += a[i]*b[i]
    return s

s = f(a,b)
