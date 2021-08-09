import os

print("compiling...")
args = ["c++", "-o main.out", "*.cpp", "-O3", "-fopenmp"]
command = " ".join(args)
os.system(command)

# n = [1, 50, 100]
n = [1, 10]
for i in n:
    print(f"executing for n = {i} particles")
    args = ["./main.out", str(i)]
    command = " ".join(args)
    os.system(command)
