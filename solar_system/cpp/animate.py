import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


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

def animate_trajectory(data, fps=500):
    n_frames = np.shape(data)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x [AU]", fontsize=14)
    ax.set_ylabel("y [AU]", fontsize=14)
    ax.set_zlabel("z [AU]", fontsize=14)
    sct, = ax.plot([], [], [], "o", markersize=2)

    def update(i):
        sct.set_data(data[i, :, 0], data[i, :, 1])
        sct.set_3d_properties(data[i, :, 2])

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1, blit=False)
    anim.save("animated_trajectory.gif", writer="ffmpeg", fps=fps)


infilename = "results/full_system.txt"
n = 10
dims = 3

R = read_data(infilename=infilename)
animate_trajectory(data=R)
