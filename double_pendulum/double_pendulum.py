import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def solve(n=10_0000, phi1_0=np.pi/2, phi2_0=-np.pi/2):
    phi1 = np.zeros(n)
    phi2 = np.zeros(n)

    phi1_dot = np.zeros(n)
    phi2_dot = np.zeros(n)
    g = 9.81
    r1 = 1
    r2 = 1
    dt = 0.001

    phi1[0] = phi1_0
    phi2[0] = phi2_0
    for i in range(n-1):
        a = g*(np.sin(phi2[i])-2*np.sin(phi1[i]))
        phi1_dot[i+1] = phi1_dot[i] + a*dt
        phi1[i+1] = phi1[i] + phi1_dot[i+1]*dt

        a = -2*g*(np.sin(phi2[i]) - np.sin(phi1[i]))
        phi2_dot[i+1] = phi2_dot[i] + a*dt
        phi2[i+1] = phi2[i] + phi2_dot[i+1]*dt

    return phi1, phi2

phi1, phi2 = solve()

r1 = np.zeros([len(phi1),2])
r2 = np.zeros([len(phi2),2])


r1[:,0] = np.sin(phi1)
r1[:, 1] = -np.cos(phi1)

r2[:,0] = np.sin(phi2)
r2[:, 1] = -np.cos(phi2)
r2 += r1

plt.plot(r1[:,0], r1[:,1])
plt.plot(r2[:,0], r2[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
