#include <cstdio>
#include <cstdlib>
#include <armadillo>

using namespace arma;
using namespace std;

int main(int argc, char const *argv[]) {
  int N = 10000;
  mat A(N,N);
  A.randu();
  printf("Computing the inverse of A...\n");
  mat B = inv(A);
  printf("Finished.\n");

  return 0;
}
