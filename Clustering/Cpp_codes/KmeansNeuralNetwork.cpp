#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "KmeansNeuralNetwork.hpp"
#include <time.h>

using namespace std;

void KmeansNeuralNetwork::ReadData(char* filename_training, char* filename_testing)
{
  FILE *fp = fopen(filename_training, "r");
  fscanf(fp, "%*s %d %*s %d", &m_Nsamples, &m_SizeOfSample);
  printf("Number of datapoints = %d\n", m_Nsamples);
  printf("Length of data = %d\n", m_SizeOfSample);
  m_DataMatrix = (double**)malloc(m_Nsamples*sizeof(double*));
  for (int i = 0; i < m_Nsamples; i++) m_DataMatrix[i] = (double*)malloc(m_SizeOfSample*sizeof(double));

  for (int i = 0; i < m_Nsamples; i++){
    for (int j = 0; j < m_SizeOfSample; j++){
      fscanf(fp, "%lf", &m_DataMatrix[i][j]);
    }
  }
  fclose(fp);

  fp = fopen(filename_testing, "r");
  fscanf(fp, "%*s %d %*s %d", &m_Ntestingdata, &m_SizeOfSample);
  printf("Number of data points for testing = %d\n", m_Ntestingdata);
  printf("Length of data = %d\n", m_SizeOfSample);

  m_testDataMatrix = (double**)malloc(m_Ntestingdata*sizeof(double*));
  for (int i = 0; i < m_Ntestingdata; i++) m_testDataMatrix[i] = (double*)malloc(m_SizeOfSample*sizeof(double));

  for (int i = 0; i < m_Ntestingdata; i++){
    for (int j = 0; j < m_SizeOfSample; j++){
      fscanf(fp, "%lf", &m_testDataMatrix[i][j]);
    }
  }
  fclose(fp);


  /* Normalize each data point. */
  double s;
  for (int i = 0; i < m_Nsamples; i++){
    s = 0.;
    for (int j = 0; j < m_SizeOfSample; j++){
      s += m_DataMatrix[i][j];
    }
    for (int j = 0; j < m_SizeOfSample; j++){
      m_DataMatrix[i][j] /= s;
    }
  }

  for (int i = 0; i < m_Ntestingdata; i++){
    s = 0.;
    for (int j = 0; j < m_SizeOfSample; j++){
      s += m_testDataMatrix[i][j];
    }
    for (int j = 0; j < m_SizeOfSample; j++){
      m_testDataMatrix[i][j] /= s;
    }
  }
}

void KmeansNeuralNetwork::InitiateModel(int k_neurons, double learning_rate, int number_of_epochs)
{
  /* Defining parameters of the model */
  m_Kneurons = k_neurons;
  m_LearningRate = learning_rate;
  m_Nepochs = number_of_epochs;

  /* Allocate memory */
  m_Weights = (double**)malloc(m_Kneurons*sizeof(double*));
  for (int i = 0; i < m_Kneurons; i++) m_Weights[i] = (double*)malloc(m_SizeOfSample*sizeof(double));
  m_Activations = (double*)calloc(m_Kneurons, sizeof(double));

  srand(time(0));
  int N = 100000000;
  double normalizing_factor = 1./((double) N);
  double u, z;
  for (int i = 0; i < m_Kneurons; i++) {
    for (int j = 0; j < m_SizeOfSample; j++) {
      u = (rand() % N)*normalizing_factor;   //Uniform numbers at Interval = (0,1)
      if (j % 2 == 0){
        z = 0.01*log(2*u);
      }
      else{
        z = -0.01*log(2*(1-u));
      }
      m_Weights[i][j] = z;
    }
  }
}

void KmeansNeuralNetwork::TrainModel()
{
  double tmp, *ptr;
  int winning_neuron;
  for (int epoch = 0; epoch < m_Nepochs; epoch++){
    printf("Epoch %d of %d\n", epoch, m_Nepochs);
    for (int sample = 0; sample < m_Nsamples; sample++){
      for (int i = 0; i < m_Kneurons; i++){
        tmp = 0;
        for (int j = 0; j < m_SizeOfSample; j++){
          tmp += m_Weights[i][j]*m_DataMatrix[sample][j];
        }
        m_Activations[i] = tmp;
      }
      winning_neuron = 0;
      for (int i = 1; i < m_Kneurons; i++){
        if (m_Activations[i] > m_Activations[winning_neuron]){
          winning_neuron = i;
        }
      }
      for (int j = 0; j < m_SizeOfSample; j++){
        m_Weights[winning_neuron][j] += m_LearningRate*(m_DataMatrix[sample][j] - m_Weights[winning_neuron][j]);
      }
    }
  }
}

void KmeansNeuralNetwork::Predict()
{
  /* This function must be updated so actual test data is provided */



  double tmp;
  int winning_neuron;
  for (int sample = 0; sample < m_Ntestingdata; sample++){
    for (int i = 0; i < m_Kneurons; i++){
      tmp = 0;
      for (int j = 0; j < m_SizeOfSample; j++){
        tmp += m_Weights[i][j]*m_testDataMatrix[sample][j];
      }
      m_Activations[i] = tmp;
    }
    winning_neuron = 0;
    for (int i = 1; i < m_Kneurons; i++){
      if (m_Activations[i] > m_Activations[winning_neuron]){
        winning_neuron = i;
      }
    }
    printf("Data point belongs to cluster %d\n", winning_neuron);
  }
}
