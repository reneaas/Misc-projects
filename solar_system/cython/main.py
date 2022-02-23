from solar_system import PySolarSystem, PySolarSystemFlat
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import pyarma as pa
import sys
from mpl_toolkits.mplot3d import Axes3D
import time 


def main():

    r0 = np.load("../data/init_pos.npy")
    v0 = np.load("../data/init_vel.npy")
    m = np.load("../data/mass.npy")

    num_particles = r0.shape[0]

    solar_system = PySolarSystem(r0=r0, v0=v0, m=m)
    num_iter = int(1e6)
    r = np.zeros(shape=(num_iter, num_particles, 3))
    start = time.perf_counter()
    for i in trange(num_iter, desc="Calculating orbits"):
        force = solar_system.get_force()
        solar_system.step(force=force, dt=0.001)
        new_pos = solar_system.get_position()
        r[i, ...] = new_pos
    end = time.perf_counter()
    timeused = end - start 
    print(f"{timeused=}")
    
    #Projection in xy-plane
    for i in range(num_particles):
        plt.plot(r[:, i, 0], r[:, i, 1])
    plt.show()

    #Full 3D plot of trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_particles):
        ax.plot(xs=r[:, i, 0], ys=r[:, i, 1], zs=r[:, i, 2])
    plt.show()


def main_flat():

    r0 = np.load("../data/init_pos.npy")
    v0 = np.load("../data/init_vel.npy")
    m = np.load("../data/mass.npy")

    num_particles = r0.shape[0]
    r0 = r0.ravel()
    v0 = v0.ravel()

    solar_system = PySolarSystemFlat(r0=r0, v0=v0, m=m)
    num_iter = int(1e6)
    r = np.zeros(shape=(num_iter, num_particles * 3))
    start = time.perf_counter()
    for i in trange(num_iter, desc="Calculating orbits"):
        force = solar_system.get_force()
        solar_system.step(force=force, dt=0.001)
        new_pos = solar_system.get_position()
        r[i, ...] = new_pos
    end = time.perf_counter()
    timeused = end - start 
    print(f"{timeused=}")
    r = r.reshape((num_iter, num_particles, 3))

    
    #Projection in xy-plane
    for i in range(num_particles):
        plt.plot(r[:, i, 0], r[:, i, 1])
    plt.show()

    #Full 3D plot of trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_particles):
        ax.plot(xs=r[:, i, 0], ys=r[:, i, 1], zs=r[:, i, 2])
    plt.show()





if __name__ == "__main__":
    main_flat()