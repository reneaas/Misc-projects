#include "rbm.hpp"
#include <armadillo>
#include <time.h>

using namespace std;

void read_mnist(arma::mat *X_train, arma::mat *X_Val, arma::mat *X_test);

int main(int argc, char const *argv[]) {
    int n_hidden = 14*14;
    int n_visible = 28*28;
    arma::mat X_train, X_val, X_test;
    read_mnist(&X_train, &X_val, &X_test);


    double eta = 0.1;
    double mom = 0.9;
    int epochs = 10;
    int batch_sz = 100;
    int nCDsteps = 25;

    RBM my_solver(n_visible, n_hidden, eta, mom);
    clock_t start = clock();
    my_solver.fit(X_train, epochs, batch_sz, nCDsteps);
    clock_t end = clock();
    double timeused = (double) (end-start)/CLOCKS_PER_SEC;
    std::cout << "timeused = " << timeused << " seconds " << std::endl;

    return 0;
}


void read_mnist(arma::mat *X_train, arma::mat *X_val, arma::mat *X_test){
    (*X_train).load("../datasets/mnist_X_train.bin");
    (*X_val).load("../datasets/mnist_X_val.bin");
    (*X_test).load("../datasets/mnist_X_test.bin");
}
