from py_solar_storm import get_B_field
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

m = 1

def get_Lorentz_force(v, B, q=1e6):
    """Computes the Lorentz force of on the particle given a
    velocity and magnetic field.

    Args:
        v (np.ndarray):
            Current velocity of particle. Shape: (3)
        B (np.ndarray):
            Current magnetic field. Shape: (3)
        q (float):
            Charge of the particle. Default: 1e6.

    Returns:
        np.ndarray of shape (3) with the computes Lorentz force.

    
    """
    return q * np.cross(v, B)

def compute_evolution(r0, v0, num_iter, dt=0.001, mass=1.):
    """Computes the time evolution of a charged particle.

    Args:
        r0 (np.ndarray or list):
            Initial position. Must be a length 3.
        v0 (np.ndarray or list):
            Initial velocity. Must be of length 3.
        num_iter (int):
            Number of iterations to evolve the system.
        dt (float):
            Step size in time.
        mass (float):
            Mass of particle. Default: 1.
        
    Returns:
        r (np.ndarray):
            Positions of the particle of shape (num_iter, 3).
        v (np.array):
            Velocities of the particle of shape (num_iter, 3).   
    """
    r = np.zeros(shape=(num_iter, 3))
    v = np.zeros(shape=(num_iter, 3))
    r[0, ...] = r0
    v[0, ...] = v0
    for i in trange(num_iter-1, desc="Computing trajectories"):
        B = get_B_field(r[i], num_results=int(1e6))
        F = get_Lorentz_force(v=v[i], B=B)
        v[i+1] = v[i] + dt * F / mass
        r[i+1] = r[i] + dt * v[i+1]
    return r, v


def plot_b_field():
    """Plots the field lines of the magnetic field of the simplified earth model"""
    num_points = 101
    y = np.linspace(-2, 2, num_points)
    z = np.copy(y)
    # B_field = np.zeros(shape=(x.shape[0], z.shape[0]))
    B_y = np.zeros(shape=[num_points, num_points])
    B_z = np.zeros_like(B_y)
    for i in trange(num_points):
        for j in range(num_points):
            r = np.array([0., y[i], z[j]])
            _, B_y[i, j], B_z[i, j] = get_B_field(r=r, num_results=int(1e5))
    
    B_y = B_y.T
    B_z = B_z.T
    plt.streamplot(y, z, B_y, B_z, density=2)

    phi = np.linspace(0, 2 * np.pi, 1001)
    x = np.cos(phi)
    y = np.sin(phi)
    plt.plot(x, y, color="red")
    plt.axis("equal")
    plt.show()



def main():
    r0 = np.random.normal(size=3)
    r0 = np.array([4., 0., 0])
    v0 = np.array([-0.05, 0.0, 0.01]) #Along x-axis slight perturbed in z direction.
    num_results = int(1e7)

    # Compute time evoliution
    num_iter = 10000
    r, v = compute_evolution(r0=r0, v0=v0, num_iter=num_iter, dt=0.01)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    ax.plot(xs=r[:, 0], ys=r[:, 1], zs=r[:, 2], color="red")
    ax.plot_surface(x, y, z, alpha=0.5)
    # plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
    # plot_b_field()
