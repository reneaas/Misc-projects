from py_solar_system import PySolarSystem
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
    r0 = r0.ravel()
    v0 = v0.ravel()

    solar_system = PySolarSystem(r0=r0, v0=v0, m=m)
    num_iter = int(1e6)
    dt = 0.001
    r = np.zeros(shape=(num_iter, num_particles * 3))
    start = time.perf_counter()
    for i in trange(num_iter, desc="Calculating orbits"):
        force = solar_system.get_force()
        solar_system.step(force=force, dt=dt)
        r[i, ...] = solar_system.get_position()
    end = time.perf_counter()
    timeused = end - start 
    print(f"{timeused=} s")
    print(f"total years simulated = {dt * num_iter}")


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

    km_per_AU = 150e6

    r_earth = r[:, 3, ...] #i = 3 is earth
    r_mars = r[:, 4, ...] #i = 4 is mars
    print(r_earth.shape)
    distance = np.linalg.norm(r_earth - r_mars, axis=1)
    print(f"{distance.shape=}")
    mean_distance = np.mean(distance)
    print(f"{mean_distance=} AU")
    mean_distance = mean_distance * km_per_AU
    print(f"{mean_distance=} km")
    print(f"Time for light to travel from mars to earth = {mean_distance / (3e5) / 60} minutes")


    min_distance = np.min(distance, axis=0) * km_per_AU / 1e6
    max_distance = np.max(distance, axis=0) * km_per_AU / 1e6
    print(f"{min_distance=} million km")
    print(f"{max_distance=} million km")
    





if __name__ == "__main__":
    main()