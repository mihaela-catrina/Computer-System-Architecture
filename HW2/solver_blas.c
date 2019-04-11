/*
 * Tema 2 ASC
 * 2019 Spring
 * Catalin Olaru / Vlad Spoiala
 */
#include "utils.h"

/*
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
//	double alpha, beta;
	// keep the result
//	double *C = calloc(N * N, sizeof(double));
//	alpha = 1.0;
//	beta = 0.0;
//	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
//			N, N, N, alpha, B, N, A, N, beta, C, N);
//
//	alpha = 1.0;
//	beta = 1.0;
//	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
//			N, N, N, alpha, A, N, B, N, beta, C, N);

	double *D = calloc(N * N, sizeof(double));
//	alpha = 1.0;
//	beta = 0.0;
//	cblas_dgemm(CblasUpper, CblasNoTrans, CblasNoTrans,
//			N, N, N, alpha, C, N, C, N, beta, D, N);

//	mkl_free(C);

	return D;
}

