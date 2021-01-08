#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "functions.h"

int main(int argc, char const *argv[]) {
	int nvisible = 28*28;
	int nhidden = 14*14;
	double **weights, *visiblebias, *hiddenbias, *visibleact, *hiddenact;
	printf("Initializing Model...\n");
	Initialize(nvisible, nhidden, &weights, &visibleact, &hiddenact, &visiblebias, &hiddenbias);


	char* filename = "MNIST_training_data.txt";
	int **data_matrix;
	int Ndatapoints, data_length;
	printf("Reading data from file...\n");
	Read_data(&Ndatapoints, &data_length, &data_matrix, filename);

	/* Train the model */
	clock_t start, end;
	printf("Training model...\n");
	start = clock();
	Contrastive_divergence(nvisible, nhidden, weights, visibleact, hiddenact, visiblebias, hiddenbias, data_matrix, Ndatapoints);
	end = clock();
	double timeused = (double) (end-start)/CLOCKS_PER_SEC;
	printf("Time used training = %lf\n", timeused);
	/*
	   for (int i = 0; i < nvisible; i++){
	   for (int j = 0; j < nhidden; j++){
	   printf("%lf ", weights[i][j]);
	   }
	   printf("\n");
	   }
	 */
	printf("Writing the parameters to file...\n");
	Write_to_file(nvisible, nhidden, weights, visiblebias, hiddenbias);
	printf("Finished.\n");

	return 0;
}
