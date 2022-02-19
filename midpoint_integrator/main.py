from integrator import integrate, integrate_mc
import numpy as np
import time

def py_midpoint(a, b, n, f):
    x = np.linspace(a, b, n)
    h = x[1] - x[0]
    x += h
    return h * np.sum(f(x[:-1]))

def pymc_integrate(a, b, n, f):
    return (b-a) * np.mean(
        f(np.random.uniform(a, b, n))
    )

a = 0
b = 1
n = 2 ** 28
start = time.perf_counter()
res = integrate(a, b, n)
end = time.perf_counter()
timeused = end - start
print(f"{timeused=}")
print(f"{res=}")


f = lambda x: np.exp(-x)
start = time.perf_counter()
res = py_midpoint(a, b, n, f)
end = time.perf_counter()
timeused = end - start
print(f"{timeused=}")
print(f"{res=}")


start = time.perf_counter()
res = integrate_mc(a, b, n)
end = time.perf_counter()
timeused = end - start
print(f"{timeused=}")
print(f"{res=}")

start = time.perf_counter()
res = pymc_integrate(a, b, n, f)
end = time.perf_counter()
timeused = end - start
print(f"{timeused=}")
print(f"{res=}")