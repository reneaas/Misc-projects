import numpy as np
import time
N = 10000
A = np.random.randint(0,1001, size=(N,N))

print("Finding the inverse matrix...")
start = time.time()
A_inv = np.linalg.inv(A)
end = time.time()
print("Finished that shit yeah...")
timeused = end-start;
print("Time used = ", timeused)