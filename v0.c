#include<stdio.h>
#include<stdlib.h>
#include <time.h>

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

knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

double randomReal(double low, double high) {
  double d;

  d = (double) rand() / ((double) RAND_MAX + 1);
  return (low + d * (high - low));
}

void main() {
    int n = 100;
    int d = 2;
    int m = 10;
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
    }

    // Create an Υ array n x d
    for(int i = 0; i < m * d; i++) {
        y_data[i] = randomReal(0, 100);
    }

    kNN(x_data, y_data, n, m, d, k);
}