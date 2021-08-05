import numpy as np
import matplotlib.pyplot as plt

infilename = "results/full_system.txt"
n = 10
dims = 3

def read_data(infilename):
    x, y, z = [], [], []
    with open(infilename, "r") as infile:
        lines = infile.readlines()
        timesteps = int(len(lines)/n)
        tmp = np.zeros([timesteps*n, dims])
        for t in range(timesteps*n):
            vals = lines[t].split()
            tmp[t,0] =float(vals[0])
            tmp[t,1] = float(vals[1])
            tmp[t,2] = float(vals[2])

    R = np.zeros([timesteps, n, dims])
    R.flat[:] = tmp.flat[:]
    return R

R = read_data(infilename=infilename)

for i in range(n):
    x = R[:,i, 0]
    y = R[:, i, 1]
    plt.plot(x, y)

plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
l = ax.plot(R[:,0,0],R[:,0,1],R[:,0,2])
for i in range(1,n):
    l = ax.plot(R[:,i,0],R[:,i,1],R[:,i,2])
    # l = ax.plot(X[:,i],Y[:,i],Z[:,i])

ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_zlabel("z [AU]")
plt.show()
