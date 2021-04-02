import numpy as np

class SpinSystem:

    def __init__(self, L):
        self.L = L
        self.n_spins = L*L
        self.spin_mat_ = 2*np.random.randint(0, 2, (L,L)) - 1

        self.cluster = np.ones(self.spin_mat_.shape)

    def spin_mat(self, i, j):
        return self.spin_mat_[self.idx(i), self.idx(j)]


    def idx(self, i):
        return (i + self.L) % self.L

    def add_to_cluster(self, i, j):
        self.cluster[self.idx(i), self.idx(j)] = -1

def wolff(spin_system, beta):
    def add_neighbours(spin_system, i, j):
        state = True
        while state:
            if spin_system.spin_mat(i,j)*spin_system.spin_mat(i, j+1) != 1\
                and spin_system.spin_mat(i,j)*spin_system.spin_mat(i, j-1) != 1\
                and spin_system.spin_mat(i,j)*spin_system.spin_mat(i+1, j) != 1\
                and spin_system.spin_mat(i,j)*spin_system.spin_mat(i-1, j1) != 1:
                state = False

            if spin_system.spin_mat(i,j)*spin_system.spin_mat(i, j+1) == 1 and np.random.uniform(0,1) <= p:
                spin_system.add_to_cluster(i, j+1)
                add_neighbours(spin_system, i, j+1)


            if spin_system.spin_mat(i,j)*spin_system.spin_mat(i, j-1) == 1 and np.random.uniform(0,1) <= p:
                spin_system.add_to_cluster(i, j-1)
                add_neighbours(spin_system, i, j-1)

            if spin_system.spin_mat(i, j)*spin_system.spin_mat(i+1, j) == 1 and np.random.uniform(0,1) <= p:
                spin_system.add_to_cluster(i+1, j)
                add_neighbours(spin_system, i+1, j)

            if spin_system.spin_mat(i, j)*spin_system.spin_mat(i-1, j) == 1 and np.random.uniform(0,1) <= p:
                spin_system.add_to_cluster(i-1, j)
                add_neighbours(spin_system, i-1, j)

    L = spin_system.L
    spin_system.cluster[:] = 1 #reset cluster matrix

    p = 1 - np.exp(-2*beta)

    #Choose starting point for the cluster.
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    spin_system.add_to_cluster(i,j)
    add_neighbours(spin_system, i, j)

    return spin_system



L = 4
spin_system = SpinSystem(L)

T = 1
beta = 1./T


spin_system = wolff(spin_system, beta)
print(spin_system.cluster)
