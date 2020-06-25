import numpy as np
import time

n = 20000
A = np.random.randn(n,n).astype('float64')
B = np.random.randn(n,n).astype('float64')


start_time = time.time()
norm = np.linalg.norm(A @ B)
end_time = time.time()
print("Took {} seconds".format(end_time-start_time))
print("norm = ", norm)
