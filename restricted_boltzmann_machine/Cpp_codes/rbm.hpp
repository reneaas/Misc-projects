#ifndef RBM_H
#define RBM_H

#include <iostream>
#include <cmath>
using namespace std;

class RBM {
private:
  int m_Nepochs;
  int m_nCDsteps;
  int m_Nvisible;
  int m_Nhidden;
  double m_LearningRate;
  double **m_Weights;
  double *m_VisibleBias;
  double *m_HiddenBias;
  double *m_VisibleActivation;
  double *m_HiddenActivation;
  int **m_DataMatrix;
  int m_NdataPoints;
  int m_DataLength;

public:
  void Initialize(int Nvisible, int Nhidden, int Nepochs, double LearningRate, int nCDsteps);
  void ReadData(char *filename);
  void ComputeVisible();
  void ComputeHidden();
  void ContrastiveDivergence();
  void WriteParametersToFile();
};

#endif
