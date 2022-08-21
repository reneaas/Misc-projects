from ising2d import PyIsing2D
import numpy as np

def main():
    L = 2
    S = 2 * np.random.randint(0, 2, size=(L, L)) - 1
    print(S)

    spin_system = PyIsing2D(spin_matrix=S, T=1)
    print(dir(spin_system))
    print(spin_system)

    E = spin_system.get_energy()
    M = spin_system.get_magnetization()
    print(f"{E = }")
    print(f"{M = }")

if __name__ == "__main__":
    main()