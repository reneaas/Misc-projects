from py_solar_system import PySolarSystem
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def main():
    num_particles = 3
    r0 = np.random.normal(size=(num_particles, 3))
    v0 = np.random.normal(size=(num_particles, 3))
    m = np.random.uniform(size=num_particles)

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

    num_iter = 100000
    r = np.zeros(shape=(num_iter, num_particles, 3))
    v = np.zeros(shape=(num_iter, num_particles, 3))
    for i in trange(num_iter):
        force = system.get_force()
        system.step(force=force, dt=0.001)
        new_pos = system.get_position()
        new_vel = system.get_velocity()
        r[i, ...] = new_pos
        v[i, ...] = new_vel
    
    plt.plot(r[:, 0, 0], r[:, 0, 1])
    plt.show()






if __name__ == "__main__":
    main()