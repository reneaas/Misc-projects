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
