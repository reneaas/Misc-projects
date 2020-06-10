import numpy as np
from time import time

N = 15000
A = np.random.normal(0,1, (N,N))
B = np.random.normal(0,1, (N,N))

start = time()
norm = np.linalg.norm(A@B)
end = time()
print("Timeused = ", end-start)
