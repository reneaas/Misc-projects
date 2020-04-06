#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include "KmeansNeuralNetwork.hpp"
#include <string>

using namespace std;

int main(int argc, char *argv[]) {
  int k_neurons = 10;
  double learning_rate = 0.1;
  int number_of_epochs = 100;
  KmeansNeuralNetwork Cluster;
  char *filename = argv[1];
  Cluster.ReadData(filename);
  Cluster.InitiateModel(k_neurons, learning_rate, number_of_epochs);
  Cluster.TrainModel();
  Cluster.Predict();
  return 0;
}
