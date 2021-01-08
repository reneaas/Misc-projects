#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include "rbm.hpp"
#include <string>
using namespace std;

void RBM::Initialize(int Nvisible, int Nhidden, int Nepochs, double LearningRate, int nCDsteps)
{
	/* Defining parameters of the model */
	m_Nvisible = Nvisible;
	m_Nhidden = Nhidden;
	m_Nepochs = Nepochs;
	m_LearningRate = LearningRate;
	m_nCDsteps = nCDsteps;

	/* Allocate memory for the necessary arrays of the model */

	m_Weights = (double**)malloc(m_Nvisible*sizeof(double*));
	for (int i = 0; i < m_Nvisible; i++) m_Weights[i] = (double*)malloc(m_Nhidden*sizeof(double));

	m_VisibleActivation = (double*)malloc(m_Nvisible*sizeof(double));
	m_VisibleBias = (double*)malloc(m_Nvisible*sizeof(double));
	m_HiddenActivation = (double*)malloc(m_Nhidden*sizeof(double));
	m_HiddenBias = (double*)malloc(m_Nhidden*sizeof(double));

	srand(time(0));
	int N = 100000000;
	double normalizing_factor = 1./((double) N);
	double u, z;
	srand(time(0));
	for (int i = 0; i < m_Nvisible; i++) {
		for (int j = 0; j < m_Nhidden; j++) {
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

	for (int i = 0; i < m_Nvisible; i++) {
		u = (rand() % N)*normalizing_factor;   //Uniform numbers at Interval = (0,1)
		if (i % 2 == 0){
			z = 0.01*log(2*u);
		}
		else{
			z = -0.01*log(2*(1-u));
		}
		m_VisibleBias[i] = z;
	}

	for (int i = 0; i < m_Nhidden; i++) {
		u = (rand() % N)*normalizing_factor;   //Uniform numbers at Interval = (0,1)
		if (i % 2 == 0){
			z = 0.01*log(2*u);
		}
		else{
			z = -0.01*log(2*(1-u));
		}
		m_HiddenBias[i] = z;
	}
}

void RBM::ReadData(char* filename)
{
	FILE *fp = fopen(filename, "r");
	fscanf(fp, "%*s %d %*s %d", &m_NdataPoints, &m_DataLength);
	printf("Number of datapoints = %d\n", m_NdataPoints);
	printf("Length of data = %d\n", m_DataLength);
	m_DataMatrix = (int**)malloc(m_NdataPoints*sizeof(int*));
	for (int i = 0; i < m_NdataPoints; i++) m_DataMatrix[i] = (int*)malloc(m_DataLength*sizeof(int));

	double tmp;
	for (int i = 0; i < m_NdataPoints; i++){
		for (int j = 0; j < m_DataLength; j++){
			fscanf(fp, "%lf", &tmp);
			if (tmp > 0.5){
				m_DataMatrix[i][j] = 1;
			}
			else{
				m_DataMatrix[i][j] = 0;
			}
		}
	}
	fclose(fp);
}

void RBM::ComputeVisible()
{
	double prob, u;
	srand(time(0));
	int rand_max = 100000000;
	double normalizing_factor = 1/((double) rand_max);
	for (int i = 0; i < m_Nvisible; i++){
		prob = 0.;
		for (int j = 0; j < m_Nhidden; j++){
			prob += m_Weights[i][j]*m_HiddenActivation[j];
		}
		prob += m_VisibleBias[i];
		prob = 1./(1 + exp(-prob));
		u = (rand() % rand_max)*normalizing_factor;
		if (prob > u){
			m_VisibleActivation[i] = 1;
		}
		else{
			m_VisibleActivation[i] = 0;
		}
	}
}

void RBM::ComputeHidden()
{
	double prob;
	for (int j = 0; j < m_Nhidden; j++){
		prob = 0.;
		for (int i = 0; i < m_Nvisible; i++){
			prob += m_Weights[i][j]*m_VisibleActivation[i];
		}
		prob += m_HiddenBias[j];
		prob = 1./(1 + exp(-prob));
		m_HiddenActivation[j] = prob;
	}
}


void RBM::ContrastiveDivergence()
{
	double **CDpos_weights, *CDpos_visiblebias, *CDpos_hiddenbias;
	double **CDneg_weights, *CDneg_visiblebias, *CDneg_hiddenbias;

	CDpos_weights = (double**)malloc(m_Nvisible*sizeof(double*));
	for (int i = 0; i < m_Nvisible; i++) CDpos_weights[i] = (double*)malloc(m_Nhidden*sizeof(double));
	CDpos_visiblebias = (double*)malloc(m_Nvisible*sizeof(double));
	CDpos_hiddenbias = (double*)malloc(m_Nhidden*sizeof(double));

	CDneg_weights = (double**)malloc(m_Nvisible*sizeof(double*));
	for (int i = 0; i < m_Nvisible; i++) CDneg_weights[i] = (double*)malloc(m_Nhidden*sizeof(double));
	CDneg_visiblebias = (double*)malloc(m_Nvisible*sizeof(double));
	CDneg_hiddenbias = (double*)malloc(m_Nhidden*sizeof(double));

	double tmp;
	for (int sample = 0; sample < m_NdataPoints; sample++){
		printf("Training sample = %d of %d\n", sample, m_NdataPoints);
		for (int epoch = 0; epoch < m_Nepochs; epoch++){
			for (int l = 0; l < m_Nvisible; l++) m_VisibleActivation[l] = (double) m_DataMatrix[sample][l];
			ComputeHidden(); //First step, compute the hidden layer.

			/* The CD-positive phase */
			for (int i = 0; i < m_Nvisible; i++){
				tmp = m_VisibleActivation[i];
				for (int j = 0; j < m_Nhidden; j++){
					CDpos_weights[i][j] = tmp*m_HiddenActivation[j];
				}
			}
			for (int i = 0; i < m_Nvisible; i++) CDpos_visiblebias[i] = m_VisibleActivation[i];
			for (int i = 0; i < m_Nhidden; i++) CDpos_hiddenbias[i] = m_HiddenActivation[i];

			/* Contrastive divergence-n*/
			for (int k = 0; k < m_nCDsteps; k++){
				ComputeVisible();
				ComputeHidden();
			}

			/* The CD-negative phase */
			for (int i = 0; i < m_Nvisible; i++){
				tmp = m_VisibleActivation[i];
				for (int j = 0; j < m_Nhidden; j++){
					CDneg_weights[i][j] = tmp*m_HiddenActivation[j];
				}
			}
			for (int i = 0; i < m_Nvisible; i++) CDneg_visiblebias[i] = m_VisibleActivation[i];
			for (int i = 0; i < m_Nhidden; i++) CDneg_hiddenbias[i] = m_HiddenActivation[i];

			for (int i = 0; i < m_Nvisible; i++){
				for (int j = 0; j < m_Nhidden; j++){
					m_Weights[i][j] += m_LearningRate*(CDpos_weights[i][j] - CDneg_weights[i][j]);
				}
			}

			for (int i = 0; i < m_Nvisible; i++) m_VisibleBias[i] += m_LearningRate*(CDpos_visiblebias[i] - CDneg_visiblebias[i]);
			for (int i = 0; i < m_Nhidden; i++) m_HiddenBias[i] += m_LearningRate*(CDpos_hiddenbias[i] - CDneg_hiddenbias[i]);
		}
		for (int l = 0; l < m_Nvisible; l++) m_VisibleActivation[l] = (double) m_DataMatrix[sample][l];
	}

	/* Free up memory that is no longer needed */
	free(CDpos_visiblebias);
	free(CDpos_hiddenbias);
	free(CDneg_hiddenbias);
	free(CDneg_visiblebias);
	for (int i = 0; i < m_Nvisible; i++){
		free(CDpos_weights[i]);
		free(CDneg_weights[i]);
	}
	free(CDpos_weights);
	free(CDneg_weights);

	/* We don't need the activations anymore since the model is trained */
	/*
	   free(m_VisibleActivation);
	   free(m_HiddenActivation);
	 */
}

void RBM::WriteParametersToFile()
{
	FILE *fp;
	fp = fopen("weights.txt", "w");
	for (int i = 0; i < m_Nvisible; i++){
		for (int j = 0; j < m_Nhidden; j++){
			fprintf(fp, "%lf ", m_Weights[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen("visiblebias.txt", "w");
	for (int i = 0; i < m_Nvisible; i++) fprintf(fp, "%lf\n", m_VisibleBias[i]);
	fclose(fp);

	fp = fopen("hiddenbias.txt", "w");
	for (int i = 0; i < m_Nhidden; i++) fprintf(fp, "%lf\n", m_HiddenBias[i]);
	fclose(fp);
}
