/*
 * Tema 2 ASC
 * 2019 Spring
 * Catalin Olaru / Vlad Spoiala
 */
#include "utils.h"

// Compute transpose of a matrix
double *transpose(double *A, int N) {
    register int i, j;
    double *B = calloc(N * N, sizeof(double));
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            B[i * N + j] = A[j * N + i];

    return B;
}

// Optimized implementation
double *my_solver(int N, double *A, double *B) {
    double *C = calloc(N * N, sizeof(double));
    double *aux = calloc(N * N, sizeof(double));
    register int k, j;
    int i, index;
    double sum = 0;
    double *At = transpose(A, N);
    double *Bt = transpose(B, N);

    // compute At x B
    for (i = 0; i < N; i++) {
        index = i * N;
        double *orig_pa = &At[index];
        for (j = 0; j < N; j++) {
            double *pa = orig_pa;
            double *pb = &Bt[j * N];
            for (k = 0; k < N; k++) {
                sum += *pa * *pb;
                pa++;
                pb++;
            }
            aux[index + j] = sum;
            sum = 0;
        }
    }

    // (At x B)t = Bt x A
    // compute sum instead of multiplication
    double *auxT = transpose(aux, N);
    for (i = 0; i < N; i++) {
        double *pa = &aux[i * N];
        double *pb = &auxT[i * N];
        for (j = 0; j < N; j++) {
            if (i <= j) {
                *pa += *pb;
            } else {
                *pa = 0;
            }

            pa++;
            pb++;
        }
    }

    // Matrix x Matrix
    // compute multiplication on lines instead of columns
    sum = 0;
    auxT = transpose(aux, N);
    for (i = 0; i < N; i++) {
        index = i * N;
        double *orig_pa = &aux[index + i];
        for (j = 0; j < N; j++) {
            double *pa = orig_pa;
            double *pb = &auxT[j * N + i];
            if (i <= j) {
                for (k = i; k <= j; k++) {
                    sum += *pa * *pb;
                    pa++;
                    pb++;
                }
                C[index + j] = sum;
                sum = 0;
            }
        }
    }

    return C;
}
