import numpy as np
import time

n = 10000
A = np.random.randn(n,n).astype('float64')
B = np.random.randn(n,n).astype('float64')
print("n = ", n)


start_time = time.time()
norm = np.linalg.norm(A @ B)
end_time = time.time()
print("Norm of C = A*B: Took {} seconds".format(end_time-start_time))

n = 5000
print("n = ", n)
A = np.random.randn(n,n).astype('float64')
start_time = time.time()
u, s, v = np.linalg.svd(A, full_matrices=True)
end_time = time.time()
print("SVD: Took {} seconds".format(end_time-start_time))


"""
start_time = time.time()
cholesky_matrix = np.linalg.cholesky(A)
end_time = time.time()
print("Cholesky: Took {} seconds".format(end_time-start_time))
"""
t = np.linspace(0,2*np.pi, 10000001)
omega = 2*np.pi*2000    #angular frequency
func_vals = np.cos(omega*t)

start_time = time.time()
F = np.fft.fft(func_vals)
end_time = time.time()
print("FFT: Took {} seconds".format(end_time-start_time))

