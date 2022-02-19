from py_solar_system import PySolarSystem
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import pyarma as pa
import sys


def main():

    r0 = np.load("../data/init_pos.npy")
    v0 = np.load("../data/init_vel.npy")
    m = np.load("../data/mass.npy")

    num_particles = r0.shape[0]

    system = PySolarSystem(r0=r0, v0=v0, m=m)
    pos = system.get_position()
    vel = system.get_velocity()
    print(f"{pos=}")
    print(f"{vel=}")

    num_objects = system.num_objects
    print(f"{num_objects=}")

    mass = system.m
    print(f"{mass=}")

    force = np.asarray(system.get_force())
    print(f"{force=}")

    system.step(force=force, dt=0.001)

    pos = system.get_position()
    vel = system.get_velocity()
    print(f"{pos=}")
    print(f"{vel=}")

    num_iter = 1000000
    r = np.zeros(shape=(num_iter, num_particles, 3))
    v = np.zeros(shape=(num_iter, num_particles, 3))
    for i in trange(num_iter):
        force = system.get_force()
        system.step(force=force, dt=0.001)
        new_pos = system.get_position()
        new_vel = system.get_velocity()
        r[i, ...] = new_pos
        v[i, ...] = new_vel
    
    for i in range(num_particles):
        plt.plot(r[:, i, 0], r[:, i, 1])
    plt.show()






if __name__ == "__main__":
    main()