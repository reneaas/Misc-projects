#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include "rbm.hpp"
#include <string>

using namespace std;

int main(int argc, char *argv[]) {
  RBM Rbm;      //Declaration of RBM object.
  int nvisible = 28*28;
  int nhidden = 10*10;
  int nepochs = 1000;
  int nCDsteps = 1;
  double LearningRate = 0.1/((double) nvisible);
  Rbm.Initialize(nvisible, nhidden, nepochs, LearningRate, nCDsteps); //Initiates up the model
  char *filename = argv[1];
  Rbm.ReadData(filename); //Reads data from the file.
  Rbm.ContrastiveDivergence(); //Trains the model.
  Rbm.WriteParametersToFile(); //Writes the parameters of the trained model to a file for further usage.
  return 0;
}
