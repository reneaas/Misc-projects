import matplotlib.pyplot as plt
import numpy as np

print("Plotting orbits...")
infilename = "computed_positions.txt"
dims = 3
with open(infilename, "r") as infile:
    lines = infile.readlines()
    timesteps = len(lines)
    number_of_objects =  int(len(lines[0].split())/3)
    r = np.zeros((timesteps, number_of_objects, 3))
    timestep = 0
    for line in lines:
        vals = line.split()
        counter = 0
        for n in range(number_of_objects):
            for k in range(dims):
                r[timestep, n, k] = float(vals[counter])
                counter += 1
        timestep += 1


for i in range(number_of_objects):
    plt.plot(r[:, i, 0], r[:, i, 1])

plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.show()
