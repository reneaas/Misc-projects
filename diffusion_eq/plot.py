import numpy as np
import matplotlib.pyplot as plt

def analytical(x,y,t):
    return np.sin(np.pi*x)*np.sin(np.pi*y)*np.exp(-2*np.pi**2*t)


x = np.linspace(0, 1, 201)
y = np.linspace(0, 1, 201)

infilename = "output.txt"
with open(infilename, "r") as infile:
    first_line = infile.readline()
    lines = infile.readlines()
    t = float(first_line.split()[-1])
    N = len(lines[1].split())
    u = np.zeros((N,N))
    for i in range(N):
        line = lines[i]
        vals = line.split()
        for j in range(N):
            u[i,j] = float(vals[j])


X, Y = np.meshgrid(x,y)
func_vals = analytical(X,Y,t)


plt.contourf(X,Y, func_vals, levels = 201)
plt.colorbar()
plt.title("Analytical solution at t = " + str(t))
plt.show()


plt.contourf(X,Y, u, levels = 201)
plt.colorbar()
plt.title("Numerical solution at t = " + str(t))
plt.show()
