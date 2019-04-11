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

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i <= j) {
                for (size_t k = 0; k < N; k++) {
                    C[i*N+j] += A[k*N + i] * B[k*N + j] + B[j*N + k]*A[k*N + i];
                }
            }
        }
    }

     int sum = 0;
     for (size_t i = 0; i < N; i++) {
     	for (size_t j = 0; j < N; j++) {
     		for (size_t k = 0; k < N; k++) {
     			sum += C[i*N + k] * C[k*N + j];
     		}
     		C[i*N + j] = sum;
     		sum = 0;
     	}
     }

	return C;
}
