#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>

#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

knnresult kNN(double * X, double * Y, int n, int m, int d, int k){

    knnresult knn_result;
    knn_result.nidx = malloc(m * k * sizeof(int));
    knn_result.ndist = malloc(m * k * sizeof(int));

    double *x_squared = malloc(n * d * sizeof(double));
    double *y_squared = malloc(m * d * sizeof(double));
    double *x_sum = malloc(n * sizeof(double));
    double *y_sum = malloc(m * sizeof(double));
    double *d_matrix = malloc(n * m * sizeof(double));

    /* sum(X.^2,2) */
    for(int i = 0; i < n * d; i++) {
        x_squared[i] = X[i] * X[i];
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < d; j++) {
            x_sum[i] += x_squared[d*i + j];
        }
    }

    /* sum(Y.^2,2) (not transposed yet) */
    for(int i = 0; i < m * d; i++) {
        y_squared[i] = Y[i] * Y[i];
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < d; j++) {
            y_sum[i] += y_squared[d*i + j];
        }
    }
    /* -2 X Y' */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d, 0, d_matrix, m);
    
    double sum = 0;
    /* (sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).') */
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            d_matrix[j + m*i] += x_sum[i] + y_sum[j];
            d_matrix[j + m*i] = sqrt(d_matrix[j + m*i]);
            sum += d_matrix[j + m*i];
        }
    }
    printf("%f\n", sum);

    // We have in our hands the D matrix in row major format
    // Next we have to search each column for the kNN

    return;
};

double randomReal(double low, double high) {
    double d;

    d = (double) rand() / ((double) RAND_MAX + 1);
    return (low + d * (high - low));
}

void main() {
    int n = 10;
    int d = 2;
    int m = 4;
    int k = 3;
    knnresult knnresult;
    knnresult.nidx = malloc(m * k * sizeof(int));
    knnresult.ndist = malloc(m * k * sizeof(int));
    knnresult.m = m;
    knnresult.k = k;

    srand(time(NULL));

    double* x_data = malloc(n * d * sizeof(double));

    double* y_data = malloc(m * d * sizeof(double));

    // Create an X array n x d
    for(int i = 0; i < n * d; i++) {
        x_data[i] = randomReal(0, 100);
        // x_data[i] = i;
    }

    // Create an Υ array n x d
    for(int i = 0; i < m * d; i++) {
        y_data[i] = randomReal(0, 100);
        // y_data[i] = i;
    }

    kNN(x_data, y_data, n, m, d, k);
}