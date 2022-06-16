from grpc import access_token_call_credentials
import numpy as np 
from tqdm import trange
import matplotlib.pyplot as plt 

class SolarSystem(object):
    def __init__(self, m, r, v):
        self.m = m # Initialize mass
        self.v = v # initialize velocities
        self.r = r # initialize positions

        self.G = 4 * np.pi ** 2

    def get_force(self, i):
        force = 0
        for j in range(self.r.shape[0]):
            if i != j:
                dr = self.r[i] - self.r[j]
                force -= self.m[j] * dr / np.linalg.norm(dr) ** 3
        force *= self.m[i] * self.G
        return force

    def get_all_forces(self):
        forces = np.zeros(shape=self.r.shape)
        for i in range(self.r.shape[0]):
            forces[i, ...] = self.get_force(i) / self.m[i]
        return forces


    def __str__(self):
        return str(self.r)


def get_euler_cromer_integrator(h, acceleration_fn):
    def euler_cromer_integrator(solar_system):
        a = acceleration_fn(solar_system)
        solar_system.v += a * h
        solar_system.r += solar_system.v * h
        return solar_system
    return euler_cromer_integrator


def main():
    r0 = np.load("../data/init_pos.npy")
    v0 = np.load("../data/init_vel.npy")
    m = np.load("../data/mass.npy")


    solar_system = SolarSystem(m, r=r0, v=v0)
    print(solar_system)

    h = 1e-2
    acceleration_fn = lambda solar_system: solar_system.get_all_forces()
    integrator = get_euler_cromer_integrator(h=h, acceleration_fn=acceleration_fn)

    num_iter = 10000
    r = np.zeros(shape=(num_iter, r0.shape[0], r0.shape[1]))
    v = np.zeros(shape=(num_iter, r0.shape[0], r0.shape[1]))
    r[0, ...] = r0
    v[0, ...] = v0
    for i in trange(num_iter - 1, desc="Computing orbits"):
        solar_system = integrator(solar_system)
        r[i+1, ...] = solar_system.r
        v[i+1, ...] = solar_system.v

    for i in range(r.shape[1]):
        plt.plot(r[:, i, 0], r[:, i, 1])
    plt.show()


    

if __name__ == "__main__":
    main()

