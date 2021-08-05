import numpy as np
import sys
import matplotlib.pyplot as plt
from time import time
import os
sys.setrecursionlimit(1_000_000)
np.random.seed(10)
# pa.pyarma_rng.set_seed(100)


class Ising2D(object):
    """
    Stores attributes of a 2D Ising model.
    """

    def __init__(self, L, temp, config = "random"):
        self.L = L
        self.temp = temp
        self.n_spins = L*L
        self.beta = 1/temp

        if config == "ordered":
            self.spin_mat_ = np.zeros([L, L])
            self.spin_mat_[:] = 1.

        elif config == "random":
            self.spin_mat_ = 2.*np.random.randint(0, 2, size=(L, L)) - 1

        else:
            self.spin_mat_ = 2.*np.random.randint(0, 2, size=(L, L)) - 1


        self._init_observables()

    def _init_observables(self):
        self.energy = 0
        self.magnetization = 0
        for i in range(self.L):
            for j in range(self.L):
                self.energy += self.spin_mat(i, j)*(self.spin_mat(i+1, j) + self.spin_mat(i, j+1))
                self.magnetization += self.spin_mat(i, j)

    def _idx(self, i):
        return (i + self.L) % self.L

    def spin_mat(self, i, j):
        return self.spin_mat_[self._idx(i), self._idx(j)]



class Wolff2D(Ising2D):
    """
    Implements the Wolff clustering sampling method to the 2D Ising model
    """

    def __init__(self, L, temp, config = "random"):
        super(Wolff2D, self).__init__(L, temp, config)

        self.cluster_ = np.zeros(self.spin_mat_.shape)
        self.prob_add = 1 - np.exp(-2*self.beta)

    def wolff(self):
        self.cluster_[:] = 1
        i = np.random.randint(0, self.L)
        j = np.random.randint(0, self.L)
        self.cluster_[i, j] = -1
        self._expand_cluster(i, j)
        self.spin_mat_ *= self.cluster_
        self.compute_magnetization_change(i, j)


    def cluster(self, i, j):
        return self.cluster_[self._idx(i), self._idx(j)]

    def _add_to_cluster(self, i, j):
        self.cluster_[self._idx(i), self._idx(j)] = -1

    def _expand_cluster(self, i, j):
        if self.spin_mat(i, j)*self.spin_mat(i+1, j) == 1 and self.cluster(i+1, j) != -1:
            if np.random.uniform() <= self.prob_add:
                self._add_to_cluster(i+1, j)
                self._expand_cluster(i+1, j)

        if self.spin_mat(i, j)*self.spin_mat(i-1, j) == 1 and self.cluster(i-1, j) != -1:
            if np.random.uniform() <= self.prob_add:
                self._add_to_cluster(i-1, j)
                self._expand_cluster(i-1, j)

        if self.spin_mat(i, j)*self.spin_mat(i, j+1) == 1 and self.cluster(i, j+1) != -1:
            if np.random.uniform() <= self.prob_add:
                self._add_to_cluster(i, j+1)
                self._expand_cluster(i, j+1)

        if self.spin_mat(i, j)*self.spin_mat(i, j-1) == 1 and self.cluster(i, j-1) != -1:
            if np.random.uniform() <= self.prob_add:
                self._add_to_cluster(i, j-1)
                self._expand_cluster(i, j-1)

    def periodic(self, i):
        return (i + self.n_spins) % self.n_spins



    def compute_magnetization_change(self, i, j):
        cluster_sites = np.where(self.cluster_ == -1)

        self.magnetization += 2*len(cluster_sites[0])*self.spin_mat(i, j)

    def get_magnetization(self):
        return self.magnetization


def monte_carlo_sim(mc_cycles, L = 16, temp = 1.0):
    spin_system = Wolff2D(L, temp)
    img = plt.imshow(spin_system.spin_mat_, cmap = "gray")
    plt.savefig(f"lattice_at_i={0}.pdf")
    plt.close()
    M_mean = 0
    for i in range(mc_cycles):
        print("i = ", i)
        spin_system.wolff()

        M_mean += abs(spin_system.get_magnetization())
        # if i % 1 == 0:
        #     # print(i)
        #     plt.imshow(spin_system.spin_mat_, cmap = "gray")
        #     plt.savefig(f"lattice_at_i={i}.pdf")
        #     plt.close()
        #     os.system(f"mv lattice_at_i={i}.pdf figures/")
            # plt.show()
    M_mean /= (mc_cycles*L*L)
    print("<|M|> = ", M_mean)

def get_magnetization_vs_temp(mc_cycles, L, num_temps):
    T = np.linspace(0.5, 3.0, num_temps)
    M = np.zeros_like(T)
    MM = np.zeros_like(T)
    for i in range(num_temps):
        current_temp = T[i]
        spin_system = Wolff2D(L, current_temp)
        M_mean = 0
        MM_mean = 0
        for j in range(mc_cycles):
            print("i = ", j)
            spin_system.wolff()
            m = spin_system.get_magnetization()
            M_mean += abs(m)
            MM_mean += m**2
        M_mean /= (mc_cycles*L*L)
        MM_mean /= (mc_cycles*L*L)
        M[i] = M_mean
        MM[i] = MM_mean

    plt.plot(T, M)
    plt.ylabel("Magnetization per spin site")
    plt.xlabel("T/J")
    plt.savefig("magnetization_per_spin_site_L=16.pdf")
    plt.close()

    plt.plot(T, MM)
    plt.ylabel("Magnetization squared per spin site")
    plt.xlabel("T/J")
    plt.savefig("magnetization_squared_per_spin_site_L=16.pdf")
    plt.close()
start = time()
# monte_carlo_sim(mc_cycles = 1000)
get_magnetization_vs_temp(500, 16, 20)
end = time()
print("timeused = ", end-start, " seconds")
