#ifndef FUNCTIONS_H
#define FUNCTIONS_H

void Initialize(int nvisible, int nhidden, double ***weights, double **visibleact, double** hiddenact, double** visiblebias, double** hiddenbias);
void Read_data(int *number_of_lines, int *length_of_datapoint, int ***test_data, char* filename);
void Contrastive_divergence(int nvisible, int nhidden, double **weights, double *visibleact, double *hiddenact, double *visiblebias, double* hiddenbias, int **dataset, int number_of_samples);
void Compute_visible(int nvisible, int nhidden, double **weights, double *visibleact, double *hiddenact, double *visiblebias);
void Compute_hidden(int nvisible, int nhidden, double **weights, double *visibleact, double *hiddenact, double *hiddenbias);
void Write_to_file(int nvisible, int nhidden, double **weights, double *visiblebias, double *hiddenbias);

#endif
