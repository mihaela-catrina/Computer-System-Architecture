/*
 * Tema 2 ASC
 * 2019 Spring
 * Catalin Olaru / Vlad Spoiala
 */
#include "utils.h"

/*
 * Add your unoptimized implementation here
 */
double* my_solver(int N, double *A, double* B) {
    double *C = calloc(N * N, sizeof(double));
    double *aux = calloc(N * N, sizeof(double));
    size_t i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i <= j) {
                for (k = 0; k < N; k++) {
                    aux[i*N+j] += A[k*N + i] * B[k*N + j] + B[k*N + i]*A[k*N + j];
                }
            }
        }
    }

     double sum = 0;
     for (i = 0; i < N; i++) {
     	for (j = 0; j < N; j++) {
     		for (k = 0; k < N; k++) {
     		    sum += aux[i*N + k] * aux[k*N + j];
     		}
     		C[i*N + j] = sum;
     		sum = 0;
     	}
     }

     return C;
}
