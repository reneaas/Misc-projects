import matplotlib.pyplot as plt
import numpy as np


N = [1, 10, 50, 100]
n = 50
n_particles = 10
dims = 3
sampling = "brute_force"
alpha = []
alpha_int = []
E_int = []
EE_int = []
E = []
EE = []
for i in range(n):
    filename = "results/interacting/mean_energy_" + sampling + "_" + str(dims) + "_" + str(n_particles) + "_" + str(i) + ".txt"
    with open(filename, "r") as infile:
        lines = infile.readlines()
        for line in lines:
            vals = line.split()
            alpha.append(float(vals[0]))
            E.append(float(vals[1]))

E = np.array(E)
plt.plot(alpha, E/n_particles, label=f"interacting; n = {n_particles}")
# plt.fill_between(alpha, E - std, E + std, color="gray", alpha=0.2)
# plt.scatter(alpha, E, label="datapoints", marker="*", color="r")
plt.xlabel("alpha")
plt.ylabel("Energy")
plt.legend()
plt.show()
