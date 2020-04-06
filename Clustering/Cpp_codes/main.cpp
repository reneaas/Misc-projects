#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include "KmeansNeuralNetwork.hpp"
#include <string>

using namespace std;

int main(int argc, char *argv[]) {
  int k_neurons = 10;
  double learning_rate = 0.01;
  int number_of_epochs = 1000;
  KmeansNeuralNetwork Cluster;
  char *filename_training = argv[1];
  char *filename_testing = argv[2];
  Cluster.ReadData(filename_training, filename_testing);
  Cluster.InitiateModel(k_neurons, learning_rate, number_of_epochs);
  Cluster.TrainModel();
  Cluster.Predict();
  return 0;
}
