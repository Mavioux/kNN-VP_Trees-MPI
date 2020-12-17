#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>
#include <mpi.h>

#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      n;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

knnresult kNN(double * X, double * Y, int n, int m, int d, int k, int rank){

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

    /* sum(Y.^2,2) (not transposed) */
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
    printf("%d %f\n",rank, sum);

    // // We have in our hands the D matrix in row major format
    // // Next we have to search each column for the kNN    
    // for(int i = 0; i < m; i++) {
    //     for(int kappa = 0; kappa < k; kappa++) {
    //         double min = INFINITY; 
    //         int index = -1;
    //         for(int j = 0; j < n; j++) {
    //             if(min > d_matrix[j * m + i]) {
    //                 min = d_matrix[j * m + i];
    //                 index = j * m + i;
    //             }
    //         }
    //         knn_result.ndist[i*k+kappa] = min;
    //         knn_result.nidx[i*k+kappa] = index;
    //         d_matrix[index] =  INFINITY;
    //     }        
    // }

    return knn_result;
}

double randomReal(double low, double high) {
    double d;

    d = (double) rand() / ((double) RAND_MAX + 1);
    return (low + d * (high - low));
}

void main() {
    int n = 100;
    int d = 2;
    int k = 3;

    knnresult knnResult;
    knnResult.nidx = malloc(n * k * sizeof(int));
    if(knnResult.nidx == NULL) {
        printf("error allocating memory\n");
        exit(0);
    }
    knnResult.ndist = malloc(n * k * sizeof(int));
    if(knnResult.ndist == NULL) {
        printf("error allocating memory\n");
        exit(0);
    }
    knnResult.n = n;
    knnResult.k = k;

    srand(time(NULL));

    double* x_data = malloc(n * d * sizeof(double));
    if(x_data == NULL) {
        printf("error allocating memory\n");
        exit(0);
    }

    // Create an X array n x d
    for(int i = 0; i < n * d; i++) {
        // x_data[i] = randomReal(0, 10);
        x_data[i] = 1;
    }

    // MPI
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);    

    int chunks = n / p;

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double* x_i_data;
    int process_n;
    int process_m;

    if(world_rank == p-1) {
        x_i_data = malloc(d * chunks + n % p);
        x_i_data = &x_data[world_rank * chunks];
        process_n = chunks + n % p;
        process_m = chunks + n % p;
    }
    else {
        x_i_data = malloc(chunks * d);
        x_i_data = &x_data[world_rank * chunks];
        process_n = chunks;
        process_m = chunks;
    }

    printf("%d: x_i_data: %f process_n: %d process_m: %d\n", world_rank, x_i_data[process_n*d-1], process_n, process_m);
    knnresult temp_knn = kNN(x_i_data, x_i_data, process_n, process_m, d, k, world_rank);

    // Finalize the MPI environment.
    MPI_Finalize();
}