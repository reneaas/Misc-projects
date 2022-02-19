#include <vector>
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char const *argv[]) {

    int num_results = 100;
    int n = 5;
    int m = 4;
    vector<mat> chain;
    chain.reserve(num_results);
    for (int i = 0; i < num_results; i++) {
        chain[i] = arma::mat(n, m).randn();
    }

    for (int i = 0; i < num_results; i++) {
        chain[i].print("chain" + to_string(i));
    }

    return 0;
}
