#ifndef KMEANSNEURALNETWORK_H
#define KMEANSNEURALNETWORK_H

using namespace std;

class KmeansNeuralNetwork
{
private:
  double *m_Activations;
  double **m_DataMatrix;
  double **m_testDataMatrix;
  int m_Kneurons;
  int m_Nsamples;
  int m_SizeOfSample;
  double *m_Neuron;
  double **m_Weights;
  int m_Nepochs;
  double m_LearningRate;
public:
  void InitiateModel(int k_neurons, double learning_rate, int number_of_epochs);
  void ReadData(char* filename);
  void TrainModel();
  void Predict();
};

#endif
