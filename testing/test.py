import numpy as np
from time import time

N = 15000

A = np.random.normal(0,1, size=(N,N))
B = np.random.normal(0,1, size=(N,N))



start = time()
C = A@B
end = time()
timeused = end-start
print("Time used on matrix multiplication:", timeused, " seconds")


start = time()
norm = np.linalg.norm(C)
end = time()
timeused = end-start
print("Time used on computing matrix norm: {} seconds".format(timeused))


start = time()
C_inv = np.linalg.inv(C)
end = time()
timeused = end-start
print("Time used on computing inverse matrix: {} seconds".format(timeused))



start = time()
C_transpose = np.transpose(C)
end = time()
timeused = end-start
print("Time used on transposing matrix: {} seconds".format(timeused))


N = 5000
D = np.random.normal(0,1,size=(N,N))

start = time()
u,v = np.linalg.eig(D)
end = time()
timeused = end-start
print("Time used on computing eigenvalues and eigenvectors: {} seconds".format(timeused))

N = 5000

A = np.random.normal(0, 1, size=(N,N))

start = time()
C = np.linalg.svd(A)
end = time()
timeused = end-start
print("Time used on SVD: {} seconds".format(timeused))
