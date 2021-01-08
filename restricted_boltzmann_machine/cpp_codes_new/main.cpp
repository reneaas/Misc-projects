#include "rbm.hpp"
#include <armadillo>

int main(int argc, char const *argv[]) {
    int n_hidden = 5;
    int n_visible = 5;

    arma::mat A = arma::mat(5,7);
    std::cout << arma::size(A)[1] << std::endl;

    int n_rows = A.n_rows;
    std::cout << n_rows << std::endl;

    RBM my_solver(n_hidden, n_visible);
    

    return 0;
}
