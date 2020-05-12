import numpy as np

N = 100000000
M = 100

a = np.linspace(0,21,N)
b = np.linspace(-22,22, N)
c = np.zeros_like(a)
d = np.zeros_like(a)

for i in range(M):
    inner_product = np.dot(a,a)

print("Computed dot product")

for i in range(M):
    length = np.linalg.norm(a)

print("computed norm")

for i in range(M):
    e = np.where(a == 12)
