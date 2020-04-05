#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void Initialize(int nvisible, int nhidden, double ***weights, double **visibleact, double **hiddenact, double **visiblebias, double **hiddenbias)
{
  /* Allocate arrays */
  *visibleact = (double*)calloc(nvisible, sizeof(double*));
  *visiblebias = (double*)calloc(nvisible, sizeof(double*));
  *hiddenact = (double*)calloc(nhidden, sizeof(double*));
  *hiddenbias = (double*)calloc(nhidden, sizeof(double*));
  *weights = (double**)calloc(nvisible, sizeof(double**));
  for (int i = 0; i < nvisible; i++) (*weights)[i] = (double*)calloc(nhidden, sizeof(double*));


  /* Initialize random weights and biases */

  srand(time(0));
  int N = 100000000;
  double N_inv = 1/((double) N);
  double u, z;
  for (int i = 0; i < nvisible; i++) {
    for (int j = 0; j < nhidden; j++) {
      u = (rand() % N)*N_inv;   //Uniform numbers at Interval = (0,1)
      //z = (2*u - 1)*0.01; //Uniform on the a small interval (-a,a)
      if (j % 2 == 0){
        z = 0.01*log(2*u);
      }
      else{
        z = -0.01*log(2*(1-u));
      }
      (*weights)[i][j] = z;
    }
  }

  for (int i = 0; i < nvisible; i++) {
    u = (rand() % N)*N_inv;   //Uniform numbers at Interval = (0,1)
    //u2 = (rand() % N)*N_inv;   //Uniform numbers at Interval = (0,1)
    //z = sqrt(-2*log(u1))*cos(u2);
    //z = (2*u - 1)*0.01;
    if (i % 2 == 0){
      z = 0.01*log(2*u);
    }
    else{
      z = -0.01*log(2*(1-u));
    }
    (*visiblebias)[i] = z;
  }

  for (int i = 0; i < nhidden; i++) {
    u = (rand() % N)*N_inv;   //Uniform numbers at Interval = (0,1)
    //u2 = (rand() % N)*N_inv;   //Uniform numbers at Interval = (0,1)
    //z = sqrt(-2*log(u1))*cos(u2);
    //z = (2*u - 1)*0.01;
    if (i % 2 == 0){
      z = 0.01*log(2*u);
    }
    else{
      z = -0.01*log(2*(1-u));
    }
    (*hiddenbias)[i] = z;
  }

}

void Compute_visible(int nvisible, int nhidden, double **weights, double *visibleact, double *hiddenact, double *visiblebias)
{
  double prob;
  double u;
  srand(time(0));
  int rand_max = 100000000;
  double normalizing_factor = 1/((double) rand_max);
  for (int i = 0; i < nvisible; i++){
    prob = 0.;
    for (int j = 0; j < nhidden; j++){
      prob += weights[i][j]*hiddenact[j];
    }
    prob += visiblebias[i];
    prob = 1./(1 + exp(-prob));
    u = (rand() % rand_max)*normalizing_factor;
    if (prob > u){
      visibleact[i] = 1;
    }
    else{
      visibleact[i] = 0;
    }
  }
}

void Compute_hidden(int nvisible, int nhidden, double **weights, double *visibleact, double *hiddenact, double *hiddenbias)
{
  double prob;
  for (int j = 0; j < nhidden; j++){
    prob = 0.;
    for (int i = 0; i < nvisible; i++){
      prob += weights[i][j]*visibleact[i];
    }
    prob += hiddenbias[j];
    prob = 1./(1 + exp(-prob));
    hiddenact[j] = prob; //Using the probabilities themselves speeds up learning.
  }
}

void Contrastive_divergence(int nvisible, int nhidden, double **weights, double *visibleact, double *hiddenact, double *visiblebias, double* hiddenbias, int **dataset, int number_of_samples)
{

  double learning_rate = 0.1/((double) nvisible);
  int nepochs = 15;
  int nCDsteps = 1;
  /* Allocate necessary arrays for the algorithm */
  double **CDpos_weights, *CDpos_visiblebias, *CDpos_hiddenbias;
  double **CDneg_weights, *CDneg_visiblebias, *CDneg_hiddenbias;
  CDpos_weights = (double**)malloc(nvisible*sizeof(double*));
  for (int i = 0; i < nvisible; i++) CDpos_weights[i] = (double*)malloc(nhidden*sizeof(double));
  CDpos_visiblebias = (double*)malloc(nvisible*sizeof(double));
  CDpos_hiddenbias = (double*)malloc(nhidden*sizeof(double));

  CDneg_weights = (double**)malloc(nvisible*sizeof(double*));
  for (int i = 0; i < nvisible; i++) CDneg_weights[i] = (double*)malloc(nhidden*sizeof(double));
  CDneg_visiblebias = (double*)malloc(nvisible*sizeof(double));
  CDneg_hiddenbias = (double*)malloc(nhidden*sizeof(double));


  double tmp;
  for (int sample = 0; sample < number_of_samples; sample++){
    printf("Training sample = %d of %d\n", sample, number_of_samples);
    for (int epoch = 0; epoch < nepochs; epoch++){
      for (int l = 0; l < nvisible; l++) visibleact[l] = (double) dataset[sample][l];
      //visibleact = (double*) dataset[sample];
      Compute_hidden(nvisible, nhidden, weights, visibleact, hiddenact, hiddenbias);

      /* The CD-positive phase */
      for (int i = 0; i < nvisible; i++){
        tmp = visibleact[i];
        for (int j = 0; j < nhidden; j++){
          CDpos_weights[i][j] = tmp*hiddenact[j];
        }
      }
      for (int i = 0; i < nvisible; i++) CDpos_visiblebias[i] = visibleact[i];
      for (int i = 0; i < nhidden; i++) CDpos_hiddenbias[i] = hiddenact[i];
      //CDpos_visiblebias = visibleact;
      //CDpos_hiddenbias = hiddenact;

      /* Contrastive divergence-n*/
      for (int k = 0; k < nCDsteps; k++){
        Compute_visible(nvisible, nhidden, weights, visibleact, hiddenact, visiblebias);
        Compute_hidden(nvisible, nhidden, weights, visibleact, hiddenact, hiddenbias);
      }

      /* The CD-negative phase */
      for (int i = 0; i < nvisible; i++){
        tmp = visibleact[i];
        for (int j = 0; j < nhidden; j++){
          CDneg_weights[i][j] = tmp*hiddenact[j];
        }
      }
      for (int i = 0; i < nvisible; i++) CDneg_visiblebias[i] = visibleact[i];
      for (int i = 0; i < nhidden; i++) CDneg_hiddenbias[i] = hiddenact[i];
      //CDneg_visiblebias = visibleact;
      //CDneg_hiddenbias = hiddenact;

      for (int i = 0; i < nvisible; i++){
        for (int j = 0; j < nhidden; j++){
          weights[i][j] += learning_rate*(CDpos_weights[i][j] - CDneg_weights[i][j]);
        }
      }

      for (int i = 0; i < nvisible; i++) visiblebias[i] += learning_rate*(CDpos_visiblebias[i] - CDneg_visiblebias[i]);
      for (int i = 0; i < nhidden; i++) hiddenbias[i] += learning_rate*(CDpos_hiddenbias[i] - CDneg_hiddenbias[i]);
    }
    for (int l = 0; l < nvisible; l++) visibleact[l] = (double) dataset[sample][l];
    //visibleact = (double*) dataset[sample];
  }
  free(CDpos_visiblebias);
  free(CDpos_hiddenbias);
  free(CDneg_hiddenbias);
  free(CDneg_visiblebias);
  for (int i = 0; i < nvisible; i++){
    free(CDpos_weights[i]);
    free(CDneg_weights[i]);
  }
  free(CDpos_weights);
  free(CDneg_weights);
}

void Write_to_file(int nvisible, int nhidden, double **weights, double *visiblebias, double *hiddenbias)
{
  FILE *fp;
  fp = fopen("weights.txt", "w");
  for (int i = 0; i < nvisible; i++){
    for (int j = 0; j < nhidden; j++){
      fprintf(fp, "%lf ", weights[i][j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  fp = fopen("visiblebias.txt", "w");
  for (int i = 0; i < nvisible; i++) fprintf(fp, "%lf\n", visiblebias[i]);
  fclose(fp);

  fp = fopen("hiddenbias.txt", "w");
  for (int i = 0; i < nhidden; i++) fprintf(fp, "%lf\n", hiddenbias[i]);
  fclose(fp);
}

void Read_data(int *number_of_lines, int *length_of_datapoint, int ***test_data, char* filename)
{

  FILE *fp = fopen(filename, "r");
  fscanf(fp, "%*s %d %*s %d", number_of_lines, length_of_datapoint);

  /* Allocate data matrix */
  *test_data = (int**)malloc(*number_of_lines*sizeof(int**));
  for (int i = 0; i < *number_of_lines; i++) (*test_data)[i] = (int*)malloc(*length_of_datapoint*sizeof(int*));

  /* Read data from file and make it binary */
  double tmp;
  for (int i = 0; i < *number_of_lines; i++){
    for (int j = 0; j < *length_of_datapoint; j++){
      fscanf(fp, "%lf", &tmp);
      //printf("tmp = %lf\n", tmp);
      if (tmp > 0.5){
        (*test_data)[i][j] = 1;
      }
      else{
        (*test_data)[i][j] = 0;
      }
    }
  }
  fclose(fp);
}
