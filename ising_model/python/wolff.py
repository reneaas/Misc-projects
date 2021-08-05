import numpy as np
import sys
import matplotlib.pyplot as plt
from time import time
import os
sys.setrecursionlimit(1_000_000)



class Ising1D(object):
    """docstring for Ising1D."""

    def __init__(self, L, temp, config = None):
        self.L = L
        self.temp = temp
        self.beta = 1./temp

        if config == "ordered":
            self.spin_mat_ = np.zeros(L)
            self.spin_mat_[:] = 1

        elif config == "random":
            self.spin_mat_ = 2.*np.random.randint(0, 2, size=L) - 1

        else:
            self.spin_mat_ = 2.*np.random.randint(0, 2, size=L) - 1

        self._init_observables()

    def _init_observables(self):
        self.magnetization = np.sum(self.spin_mat_)

    def _idx(self, i):
        return (i + self.L) % self.L

    def spin_mat(self, i):
        return self.spin_mat_[self._idx(i)]


class Wolff1D(Ising1D):
    """docstring for Wolff1D."""

    def __init__(self, L, temp, config = "random"):
        super(Wolff1D, self).__init__(L, temp, config)

        self.cluster_ = np.zeros(self.spin_mat_.shape)
        self.prob_add = 1 - np.exp(-2*self.beta)


    def wolff(self):
        self.cluster_[:] = 1
        i = np.random.randint(0, self.L)
        print(i)
        self.cluster_[i] = -1
        self._expand_cluster(i)
        self.spin_mat_ *= self.cluster_
        self.compute_magnetization_change(i)


    def _expand_cluster(self, i):
        if self.spin_mat(i)*self.spin_mat(i+1) == 1 and self.cluster(i+1) != -1:
            if np.random.uniform() <= self.prob_add:
                self._add_to_cluster(i+1)
                self._expand_cluster(i+1)

        if self.spin_mat(i)*self.spin_mat(i-1) == 1 and self.cluster(i-1) != -1:
            if np.random.uniform() <= self.prob_add:
                self._add_to_cluster(i-1)
                self._expand_cluster(i-1)

    def _add_to_cluster(self, i):
        self.cluster_[self._idx(i)] = -1

    def cluster(self, i):
        return self.cluster_[self._idx(i)]

    def compute_magnetization_change(self, i):
        cluster_sites = np.where(self.cluster_ == -1)
        self.magnetization += 2*len(cluster_sites)*self.spin_mat(i)

    def get_magnetization(self):
        return self.magnetization




def monte_carlo_sim(mc_cycles, L = 16, temp = 1.0):
    spin_system = Wolff1D(L, temp)
    M_mean = 0
    for i in range(mc_cycles):
        # print("i = ", i)
        spin_system.wolff()
        M_mean += spin_system.get_magnetization()

    M_mean /= (mc_cycles*L)
    print("<M> = ", M_mean)




start = time()
monte_carlo_sim(mc_cycles = 500)
end = time()
print("timeused = ", end-start, " seconds")
