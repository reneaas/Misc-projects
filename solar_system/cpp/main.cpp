#include "solarsystem.hpp"


#include <armadillo>
#include <cstdio>
#include <time.h>

void read_init_cond(char *, arma::mat*, arma::mat*, int n);
void read_mass(char*, arma::vec*, int n);

int main(int argc, char const *argv[]) {
    arma::vec mass;
    mass.load("../data/mass.bin");
    arma::mat init_pos, init_vel;
    init_pos.load("../data/pos.bin");
    init_vel.load("../data/vel.bin");


    int n = mass.n_elem; //number of celestial bodies.
    int timesteps = 1e6;
    double step_sz = 0.001;
    std::string outfilename = "results/full_system.txt";
    SolarSystem my_solver(n, timesteps,step_sz);
    my_solver.init_data(init_pos, init_vel, mass);
    // SolarSystem my_solver(timesteps, step_sz);
    clock_t start = clock();
    my_solver.euler_cromer();
    clock_t end = clock();
    double timeused = (double) (end-start)/CLOCKS_PER_SEC;
    std::cout << "timeused = " << timeused << " seconds " << std::endl;
    my_solver.write_to_file(outfilename);
    return 0;
}

void read_init_cond(char *filename, arma::mat *pos, arma::mat *vel, int n){
    (*pos) = arma::mat(3, n);
    (*vel) = arma::mat(3, n);
    FILE *fp;
    fp = fopen(filename, "r");
    for (int i = 0; i < n; i++){
        int tmp = fscanf(fp, "%lf %lf %lf %lf %lf %lf", &(*pos)(0,i), &(*pos)(1,i), &(*pos)(2,i),
                                                &(*vel)(0,i), &(*vel)(1,i), &(*vel)(2,i));
    }
    fclose(fp);
}

void read_mass(char *filename, arma::vec *mass, int n){
    (*mass) = arma::vec(n);
    FILE *fp = fopen(filename, "r");
    for (int i = 0; i < n; i++){
      int tmp = fscanf(fp, "%lf", &(*mass)(i));
    }
    fclose(fp);
}
