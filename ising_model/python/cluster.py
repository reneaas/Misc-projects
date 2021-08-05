import numpy as np
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(10000)

# np.random.seed(15)

L = 20


def add_to_cluster(cluster, i, j):
    cluster[idx(i), idx(j)] = -1


#Probability p must be be changed to p = 1-np.exp(-2*beta).
def get_cluster(cluster, spin_matrix, i, j, p = 1-np.exp(-2)):
    if spin_matrix[idx(i),idx(j)]*spin_matrix[idx(i+1), idx(j)] == 1 and cluster[idx(i+1), idx(j)] != -1:
        if np.random.uniform(0,1) <= p:
            add_to_cluster(cluster, i+1, j)
            get_cluster(cluster, spin_matrix, i+1, j)

    if spin_matrix[idx(i),idx(j)]*spin_matrix[idx(i-1), idx(j)] == 1 and cluster[idx(i-1), idx(j)] != -1:
        if np.random.uniform(0,1) <= p:
            add_to_cluster(cluster, i-1, j)
            get_cluster(cluster, spin_matrix, i-1, j)

    if spin_matrix[idx(i),idx(j)]*spin_matrix[idx(i), idx(j+1)] == 1 and cluster[idx(i), idx(j+1)] != -1:
        if np.random.uniform(0,1) <= p:
            add_to_cluster(cluster, i, j+1)
            get_cluster(cluster, spin_matrix, i, j+1)

    if spin_matrix[idx(i),idx(j)]*spin_matrix[idx(i), idx(j-1)] == 1 and cluster[idx(i), idx(j-1)] != -1:
        if np.random.uniform(0,1) <= p:
            add_to_cluster(cluster, i, j-1)
            get_cluster(cluster, spin_matrix, i, j-1)

def energy_change(spin_matrix, cluster):
    cluster_sites = np.array(np.where(cluster == -1)).T
    dE = 0
    for x, y in cluster_sites:
        if not ([idx(x+1), y] == cluster_sites).all(axis=1).any():
            dE -= 2*spin_matrix[idx(x), idx(y)]*spin_matrix[idx(x+1), idx(y)]

        if not ([idx(x-1), y] == cluster_sites).all(axis=1).any():
            dE -= 2*spin_matrix[idx(x), idx(y)]*spin_matrix[idx(x-1), idx(y)]

        if not ([x, idx(y+1)] == cluster_sites).all(axis=1).any():
            dE -= 2*spin_matrix[idx(x), idx(y)]*spin_matrix[idx(x), idx(y+1)]

        if not ([x, idx(y-1)] == cluster_sites).all(axis=1).any():
            dE -= 2*spin_matrix[idx(x), idx(y)]*spin_matrix[idx(x), idx(y-1)]
    return dE

def wolff(spin_matrix):
    cluster = np.ones(spin_matrix.shape)
    i = np.random.randint(0, spin_matrix.shape[0])
    j = np.random.randint(0, spin_matrix.shape[1])
    cluster[i, j] = -1
    get_cluster(cluster, spin_matrix, i, j)
    spin_matrix = spin_matrix*cluster
    dE = energy_change(spin_matrix, cluster)
    print(dE)

    return spin_matrix


def idx(i):
    return (i + L) % L

spin_matrix = 2*np.random.randint(0,2, size=(L,L)) - 1
idx_ = np.array([int(i) for i in range(20, 40)])
spin_matrix[int(0.1*L):int(0.8*L), int(0.1*L):int(0.8*L)] = 1
tmp = spin_matrix
plt.imshow(spin_matrix, cmap="gray")
plt.show()
spin_matrix = wolff(spin_matrix)
# plt.imshow(spin_matrix, cmap="gray")
# plt.show()
