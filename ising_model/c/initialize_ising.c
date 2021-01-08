#include <stdlib.h>

void initialize_ising(char **spin_matrix, char **idx, int L, double *E, double *M)
{
    //Allocate memory
    (*spin_matrix) = (char*)calloc(L*L, sizeof(char));
    (*idx) = (char*)calloc(L+2, sizeof(char));

    //Set up index array
    (*idx)[0] = L-1;
    (*idx)[L+1] = 0;
    for (int l = 1; l <= L; l++){
        (*idx)[l] = l-1;
    }

    //Set up spin matrix in T = infty state.
    for (int i = 0; i < L*L; i++){
        (*spin_matrix)[i] = 2*(rand() % 2) - 1;
    }

    //Compute initial energy and magnetization
    (*E) = 0.;
    (*M) = 0.;
    for (int i = 1; i <= L; i++){
        for (int j = 1; j <= L; j++){
            (*E) -= (*spin_matrix)[(*idx)[i]*L + (*idx)[j]]*((*spin_matrix)[(*idx)[i+1]*L + (*idx)[j]] + (*spin_matrix)[(*idx)[i]*L + (*idx)[j+1]]);
            (*M) += (*spin_matrix)[(*idx)[i]*L + (*idx)[j]];
        }
    }
}
