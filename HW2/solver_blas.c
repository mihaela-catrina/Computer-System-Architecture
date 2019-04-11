/*
 * Tema 2 ASC
 * 2019 Spring
 * Catalin Olaru / Vlad Spoiala
 */
#include "utils.h"
#include "cblas.h"
/*
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B) {
	double alpha, beta;
	
	double *C = calloc(N * N, sizeof(double));
	alpha = 1.0;
	beta = 0.0;
	cblas_dsyr2k(CblasRowMajor, CblasUpper, CblasTrans, N, N, alpha, A, N, B, N, beta, C, N);

	double *D = calloc(N * N, sizeof(double));
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, C, N, C, N, beta, D, N);

	return D;
}

