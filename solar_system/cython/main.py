from py_solar_system import PySolarSystem
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import time



def main():
    r0 = np.load("../data/init_pos.npy")
    v0 = np.load("../data/init_vel.npy")
    m = np.load("../data/mass.npy")


    num_particles = r0.shape[0]
    r0 = r0.ravel()
    v0 = v0.ravel()

    # num_particles = 100
    # r0 = np.random.normal(size=(num_particles, 3), loc=0., scale=1.).ravel()
    # v0 = np.random.normal(size=(num_particles, 3), loc=0., scale=0.01).ravel()
    # m = np.random.uniform(low=1e-6, high=1, size=(num_particles,))


    solar_system = PySolarSystem(r0=r0, v0=v0, m=m)
    num_iter = int(1e7)
    dt = 0.000001
    r = np.zeros(shape=(num_iter, num_particles * 3))
    start = time.perf_counter()
    for i in trange(num_iter, desc="Calculating orbits"):
        force = solar_system.get_force()
        solar_system.step(force=force, dt=dt)
        r[i, ...] = solar_system.get_position()
    end = time.perf_counter()
    timeused = end - start 
    print(f"{timeused = } s")
    print(f"total years simulated = {dt * num_iter}")


    r = r.reshape((num_iter, num_particles, 3))
    #Projection in xy-plane
    for i in range(num_particles):
        plt.plot(r[:, i, 0], r[:, i, 1])
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.show()

    #Full 3D plot of trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_particles):
        ax.plot(xs=r[:, i, 0], ys=r[:, i, 1], zs=r[:, i, 2])
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.show()

    





if __name__ == "__main__":
    main()
