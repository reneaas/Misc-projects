import numpy as np


class SolarSystem(object):
    """
    mass: shape [num_objects]
    init_vel: shape [num_objects, dims]
    init_pos: shape [num_objects, dims]
    timesteps: (int) number of timesteps
    dt: (double) steplength
    dims: (int) number of physical dimensions.
    """

    def __init__(self, mass, init_vel, init_pos, timesteps, dt = 0.0001, dims  = 3):
        super(SolarSystem, self).__init__()

        self.num_objects = len(mass)
        self.mass = mass
        self.timesteps = timesteps
        self.dims = dims #Defaults to 3D simulation.

        self.pos = np.zeros([self.timesteps, self.num_objects, self.dims])
        self.vel = np.zeros_like(self.pos)
        self.time = np.zeros(timesteps)

        self.acc = np.zeros([self.num_objects, self.dims])
        self.acc_old = np.zeros_like(self.acc)

        self.pos[0, :] = init_pos[:]
        self.vel[0, :] = init_vel[:]

        self.G = 4*pi**2

    def compute_acc(self, t):
        self.acc, self.acc_old = self.acc_old, self.acc
        self.acc[:] = 0.
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if i != j:
                    diff = self.pos[t, i] - self.pos[t, j]
                    self.acc[i, :] += diff/np.linalg.norm(diff)**3
        self.acc *= self.G


class EulerCromer(object):

    def __init__(self, dt):
        self.dt = dt

        return None

    def forward(self, vel, pos, acc):
        vel = vel + acc*self.dt
        pos = pos + vel*self.dt
        return vel, pos


def compute_acc():
    return None


def run_simulation(timesteps = int(1e6)):
    mass = np.load("data/mass.npy")
    init_vel = np.load("data/vel.npy")
    init_pos = np.load("data/pos.npy")

    system = SolarSystem(mass, init_vel, init_pos, timesteps)
    solver = EulerCromer(dt = system.dt)

    for i in range(timesteps):
        system.vel[i+1] = solver.forward()
