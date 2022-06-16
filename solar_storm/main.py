from py_solar_storm import get_B_field
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def get_lorentz_force(v, B, q=1e9):
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
        np.ndarray of shape (3) with the computed Lorentz force.
    """
    return q * np.cross(v, B)



def get_rk4_integrator(h, f):
    """Constructs and returns a RK4 integrator.

        Args:
            h (float):
                Step size in time.
            f (Callable):
                Python callable with the semantics f(t, *x).
                Plays the role of the time derivative of an arbitrary 
                number of functions *x.

        Returns:
            A python callable that integrates the arbitrary functions *x
            according to f(t, *x) one time step h.
    """
    def rk4_integrator(t, *x):
        k0 = f(t, *x)

        args = [x_i + 0.5 * k * h for x_i, k in zip(x, k0)]
        k1 = f(t, *args)

        args = [x_i + 0.5 * k * h for x_i, k in zip(x, k1)]
        k2 = f(t, *args)

        args = [x_i + k * h for x_i, k in zip(x, k2)]
        k3 = f(t, *args)
        
        x_next = [
            x_i + (h / 6) * (i + 2 * j + 2 * k + l) for x_i, i, j, k, l in zip(x, k0, k1, k2, k3)
        ]
        return x_next
    return rk4_integrator


# The signature of the returned function is necessary, even if `t` and `r` are unused.
def get_velocity_fn():
    def velocity_fn(t, r, v):
        return v
    return velocity_fn

def get_acceleration_fn(q, m, num_results):
    def acceleration_fn(t, r, v):
        magnetic_field = get_B_field(r=r, num_results=num_results)
        force = get_lorentz_force(v=v, B=magnetic_field, q=q)
        return force / m
    return acceleration_fn



def plot_b_field():
    """Plots the field lines of the magnetic field of the simplified earth model"""
    num_points = 101
    y = np.linspace(-2, 2, num_points)
    z = np.copy(y)
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
    r0 = np.array([2., 0., 0])
    v0 = np.array([-1, 0.0, 4]) #Along x-axis slight perturbed in z direction.
    num_results = int(1e6)

    # Compute time evolution
    num_iter = 10000
    r = np.zeros(shape=(num_iter, 3))
    v = np.zeros(shape=(num_iter, 3))
    r[0, ...] = r0
    v[0, ...] = v0
    h = 0.001
    velocity_fn = get_velocity_fn()
    acceleration_fn = get_acceleration_fn(q=1e9, m=1, num_results=int(1e5))
    f = lambda t, r, v: (velocity_fn(t, r, v), acceleration_fn(t, r, v))
    integrator = get_rk4_integrator(h=h, f=f)


    t = 0
    for i in trange(num_iter - 1, desc="Computing time evolution"):
        r[i+1, ...], v[i+1, ...] = integrator(t, r[i], v[i])
        t += h

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
    # plt.savefig("van_allen_belt_2.pdf")
    plt.show()

if __name__ == "__main__":
    # plot_b_field()
    main()