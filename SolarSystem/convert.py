import numpy as np
filename = "masses_old.txt"
masses = []

with open(filename, "r") as infile:
    lines = infile.readlines()
    for line in lines:
        vals = line.split()
        masses.append(float(vals[0]))

masses = [m/masses[0] for m in masses]
outfilename = "masses.txt"
with open(outfilename, "w") as outfile:
    for m in masses:
        outfile.write(str(m))
        outfile.write("\n")

print(len(masses))


filename = "velocities_old.txt"
vel = np.zeros((10, 3));

with open(filename, "r") as infile:
    lines = infile.readlines()
    for i in range(len(lines)):
        vals = lines[i].split()
        vel[i,0] = float(vals[0])
        vel[i,1] = float(vals[1])
        vel[i,2] = float(vals[2])

outfilename = "velocities.txt"
conversion_factor = 365.25
vel = vel*conversion_factor
print(vel)
with open(outfilename, "w") as outfile:
    for i in range(10):
        outfile.write(str(vel[i,0]) + " " + str(vel[i,1]) + " " + str(vel[i,2]))
        outfile.write("\n")
