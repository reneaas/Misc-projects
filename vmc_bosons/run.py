import os

print("compiling...")
args = ["g++-10", "-o main.out", "*.cpp", "-Ofast", "-fopenmp", "-larmadillo", "-march=native"]
command = " ".join(args)
os.system(command)

# n = [1, 50, 100]
n = [1]
for i in n:
    print(f"executing for n = {i} particles")
    args = ["./main.out", str(i)]
    command = " ".join(args)
    os.system(command)
