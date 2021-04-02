import numpy as np
import numba


def idx(i, L):
    return (i) % L

class SpinSystem:
    def __init__(self, L, config):
        self.L = L
        self.n_spins = L*L

        if config == "random":
            self.spin_mat = 2*np.random.randint(0,2, size=(L,L)) - 1
        elif config == "ordered":
            self.spin_mat = np.ones((L, L))
        else:
            self.spin_mat = np.ones((L,L))

        self.energy = 0
        self.magnetization = 0
        for i in range(L):
            for j in range(L):
                self.energy -= self.spin_mat[idx(i, self.L), idx(j, self.L)]*(self.spin_mat[idx(i, self.L), idx(j+1, self.L)] + self.spin_mat[idx(i+1, self.L), idx(j, self.L)])
                self.magnetization += self.spin_mat[idx(i, self.L), idx(j, self.L)]


def metropolis(system, boltzmann):
    n_spins = system.n_spins
    L = system.L
    for i in range(n_spins):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        dE = 2*system.spin_mat[idx(i, L), idx(j, L)] \
                *(system.spin_mat[idx(i+1, L), idx(j, L)] \
                + system.spin_mat[idx(i-1, L), idx(j, L)] \
                + system.spin_mat[idx(i,L), idx(j+1, L)] \
                + system.spin_mat[idx(i, L), idx(j-1, L)])

        if dE <= 0:
            system.spin_mat[idx(i, L), idx(j, L)] *= -1
            system.energy += dE
            system.magnetization += 2*system.spin_mat[idx(i, L), idx(j, L)]
        elif np.random.uniform(0,1,size=1) <= boltzmann[int(dE) + 8]:
            system.spin_mat[idx(i, L), idx(j, L)] *= -1
            system.energy += dE
            system.magnetization += 2*system.spin_mat[idx(i, L), idx(j, L)]

    return system

def monte_carlo(mc_cycles, L, temp, therm_cycles = 100):
    beta = 1./temp
    boltzmann = np.zeros(17)
    for i in range(0,17,4):
        boltzmann[i] = np.exp(-beta*(i-8))

    system = SpinSystem(L, "ordered")

    for i in range(therm_cycles):
        system = metropolis(system, boltzmann)

    mean_energy = 0
    for i in range(mc_cycles):
        system = metropolis(system, boltzmann)
        mean_energy += system.energy

    print(mean_energy/(system.n_spins*mc_cycles))



monte_carlo(mc_cycles = int(1e5), L = 2, temp = 1.0)
