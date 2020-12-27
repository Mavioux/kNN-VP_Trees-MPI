#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif

//geia

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
    knn_result.ndist = malloc(m * k * sizeof(double));

    double *x_squared = malloc(n * d * sizeof(double));
    double *y_squared = malloc(m * d * sizeof(double));
    double *x_sum = malloc(n * sizeof(double));
    double *y_sum = malloc(m * sizeof(double));
    double *d_matrix = malloc(n * m * sizeof(double));

    /* sum(X.^2,2) */
    // printf("x_squared\n");
    for(int i = 0; i < n * d; i++) {
        x_squared[i] = X[i] * X[i];
        // printf("%d %f\n",i, x_squared[i]);
    }

    // printf("x_sum\n");
    for(int i = 0; i < n; i++) {
        double temp = 0;
        for(int j = 0; j < d; j++) {
            temp += x_squared[d*i + j];
            // printf("x_squared[%d]:  %f\n", d*i + j, x_squared[d*i + j]);
            // printf("temp: %f\n", temp);
        }
        x_sum[i] = temp;
        // printf("%d: %f\n", i, x_sum[i]);
    }

    /* sum(Y.^2,2) (not transposed) */
    // printf("x_squared\n");
    for(int i = 0; i < m * d; i++) {
        y_squared[i] = Y[i] * Y[i];
        // printf("%d %f\n",i, y_squared[i]);
    }

    // printf("y_sum\n");
    for(int i = 0; i < m; i++) {
        double temp = 0;
        for(int j = 0; j < d; j++) {
            temp += y_squared[d*i + j];
        }
        y_sum[i] = temp;
        // printf("%d %f\n",i, y_sum[i]);
        
    }

    /* -2 X Y' */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d, 0, d_matrix, m);
    // for(int i = 0; i < n; i++) {
    //     for(int j = 0; j < m; j++) {
    //         printf("d_matrix[%d]: %f\n", j + m*i, d_matrix[j + m*i]);
    //     }
    // }

    
    double sum = 0;
    /* (sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).') */
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            // printf("d_matrix[%d]: %f\n", j + m*i, d_matrix[j + m*i]);
            // printf("x_sum[%d]: %f\n", i, x_sum[i]);
            // printf("y_sum[%d]: %f\n", j, y_sum[j]);
            d_matrix[j + m*i] += x_sum[i] + y_sum[j];
            d_matrix[j + m*i] = sqrt(d_matrix[j + m*i]);
            sum += d_matrix[j + m*i];
        }
    }
    // printf("%d %f\n",rank, sum);

    // We have in our hands the D matrix in row major format
    // Next we have to search each column for the kNN    
    for(int i = 0; i < m; i++) {
        for(int kappa = 0; kappa < k; kappa++) {
            double min = INFINITY; 
            int index = -1;
            for(int j = 0; j < n; j++) {
                if(min > d_matrix[j * m + i]) {
                    min = d_matrix[j * m + i];
                    index = j * m + i;
                }
            }
            knn_result.ndist[kappa*m + i] = min;
            knn_result.nidx[kappa*m + i] = index;
            // printf("%d %f\n", knn_result.nidx[kappa*m + i], knn_result.ndist[kappa*m + i]);
            d_matrix[index] =  INFINITY;
        }        
    }
    return knn_result;
}

double randomReal(double low, double high) {
    double d;

    d = (double) rand() / ((double) RAND_MAX + 1);
    return (low + d * (high - low));
}

void main() {
    // MPI
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);    

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int n = 20;
    int d = 2;
    int k = 3;

    int chunks = n / p;
    double* x_i_data;
    double* y_i_send_data;
    double* y_i_receive_data;
    int process_n;
    int process_m;
    int flag = -1;
    int knn_counter = 0;

    knnresult sub_knnresult;

    if(world_rank == 0) {
        srand(time(NULL));

        double* x_data = (double *)malloc(n * d * sizeof(double));
        if(x_data == NULL) {
            printf("error allocating memory\n");
            exit(0);
        }

        // Create an X array n x d
        for(int i = 0; i < n * d; i++) {
            // x_data[i] = randomReal(0, 10);
            x_data[i] = i;
        }

        if(p > 1) {
            //Broadcast to all other processes
            for(int i = 1; i < p-1; i++) {
                // MPI_Send to each process the appropriate array
                MPI_Send(&x_data[i * chunks * d], chunks * d, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            MPI_Send(&x_data[(p-1) * chunks * d], (chunks + n % d) * d, MPI_DOUBLE, p-1, 0, MPI_COMM_WORLD);
        }

        // After sending the message and receiving confirmation we are ready to call kNN on our data
        //Initialize variables for zero process
        process_n = chunks;
        process_m = chunks;
        x_i_data = malloc(process_m * d * sizeof(double));
        x_i_data = x_data;

        sub_knnresult.k = k;
        sub_knnresult.n = process_m;
        sub_knnresult.ndist = malloc(process_m * k * sizeof(double));
        sub_knnresult.nidx = malloc(process_m * k * sizeof(int));

        // IF WE UNCOMMENT THIS WE CALCULATE WITH OURSELVES TWICE
        // sub_knnresult = kNN(x_i_data, x_i_data, process_n, process_m, d, k, world_rank);
    }
    else if (world_rank == p - 1)
    {
        //Initialize variables for this process
        process_n = (chunks + n % d);
        process_m = (chunks + n % d);
        x_i_data = malloc(process_m * d * sizeof(double));

        // First receive from the mother process
        MPI_Recv(x_i_data, process_m * d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printf("%d %f\n",world_rank, x_i_data[0]);

        sub_knnresult.k = k;
        sub_knnresult.n = process_m;
        sub_knnresult.ndist = malloc(process_m * k * sizeof(double));
        sub_knnresult.nidx = malloc(process_m * k * sizeof(int));

        // IF WE UNCOMMENT THIS WE CALCULATE WITH OURSELVES TWICE
        // // After receiving the message we are ready to call kNN on our data
        // sub_knnresult = kNN(x_i_data, x_i_data, process_n, process_m, d, k, world_rank);       

    }
    else
    {
        //Initialize variables for this process
        process_n = chunks;
        process_m = chunks;
        x_i_data = malloc(process_m * d * sizeof(double));

        // First receive from the mother process
        MPI_Recv(x_i_data, process_m * d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printf("%d %f\n", world_rank, x_i_data[0]);

        sub_knnresult.k = k;
        sub_knnresult.n = process_m;
        sub_knnresult.ndist = malloc(process_m * k * sizeof(double));
        sub_knnresult.nidx = malloc(process_m * k * sizeof(int));

        // IF WE UNCOMMENT THIS WE CALCULATE WITH OURSELVES TWICE
        // // After receiving the message we are ready to call kNN on our data
        // sub_knnresult = kNN(x_i_data, x_i_data, process_n, process_m, d, k, world_rank);
    }

    // Every process has processed its own x_i_data, now they have to pass around data to each other in a ring mode
    // That has to repeat until each process has processed all possible data aka p times

    // For the first iteration set the y_i_send_data array that will be passed
    process_n = chunks;
    y_i_send_data = malloc(process_n * d * sizeof(double));
    //REALLY IMPORTANT NOT TO SET THE POINTERS EQUAL HERE
    memcpy(y_i_send_data, x_i_data, process_m * d * sizeof(double));

    for(int i = 0; i < p; i ++) {
        // Process zero sends first then waits for the last process
        if(world_rank != 0) {
            process_n = chunks;
            y_i_receive_data = malloc(process_n * d * sizeof(double));
            MPI_Recv(y_i_receive_data, process_n * d, MPI_DOUBLE, world_rank - 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        
        MPI_Send(y_i_send_data, process_n * d, MPI_DOUBLE, (world_rank + 1) % p, i, MPI_COMM_WORLD);
        if(world_rank != 0) {
            // After sending the previous y data that the process holds, in the next iteration, we want to send the new data we just received
            memcpy(y_i_send_data, y_i_receive_data, process_m * d * sizeof(double));
        }
        if(world_rank == 0) {
            process_m = chunks;
            y_i_receive_data = malloc(process_n * d * sizeof(double));
            MPI_Recv(y_i_receive_data, process_n * d, MPI_DOUBLE, p - 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // After sending the previous y data that the process holds, in the next iteration, we want to send the new data we just received
            memcpy(y_i_send_data, y_i_receive_data, process_m * d * sizeof(double));
            // // Check that the copy is right
            // for(int i = 0; i < process_m * d; i++) {
            //     printf("%f\n", y_i_send_data[i]);
            // }
        }

        knnresult temp_knnresult;
        temp_knnresult.ndist = malloc(process_m * k * sizeof(double));
        temp_knnresult.nidx = malloc(process_m * k * sizeof(int));

        // if(world_rank == 1) {
        //     printf("x_i_data\n");
        //     for(int i = 0; i < process_m * d; i++) {
        //         printf("%f\n", x_i_data[i]);
        //     }
        //     printf("y_i_receive_data\n");
        //     for(int i = 0; i < process_m * d; i++) {
        //         printf("%f\n", y_i_receive_data[i]);
        //     }
        // }
        
        if(!knn_counter) {
            // First call of knn so we store the result in the permanent sub_knnresult
            sub_knnresult = kNN(y_i_receive_data, x_i_data, process_n, process_m, d, k, world_rank);
            knn_counter++;
        }
        else {
            temp_knnresult = kNN(y_i_receive_data, x_i_data, process_n, process_m, d, k, world_rank);
            // We have the sub_knnresult from the previous comparisons and the temp_knnresult from the last knn operation
            // We have to store in the sub_knnresult the smallest values from both
            for(int i = 0; i < process_m; i++) {
                double temp_dist[k];
                int temp_index[k];    
                int kappa_one = 0;
                int kappa_two = 0;        
                for(int kappa = 0; kappa < k; kappa++) {
                    if(temp_knnresult.ndist[kappa_one*process_m + i] < sub_knnresult.ndist[kappa_two*process_m + i]) {
                        // if(world_rank == 1)
                        //     printf("kappa_one*process_m + i: %d\n", kappa_one*process_m + i);
                        temp_dist[kappa] = temp_knnresult.ndist[kappa_one*process_m + i];
                        temp_index[kappa] = temp_knnresult.nidx[kappa_one*process_m + i];
                        kappa_one++;
                    }
                    else
                    {
                        // if(world_rank == 1)
                        //     printf("kappa_two*process_m + i: %d\n", kappa_two*process_m + i);
                        temp_dist[kappa] =  sub_knnresult.ndist[kappa_two*process_m + i];
                        temp_index[kappa] =  sub_knnresult.nidx[kappa_two*process_m + i];
                        kappa_two++;
                    }
                    // if(world_rank == 1) {
                    //     printf("kappa_one: %d\n", kappa_one);
                    //     printf("kappa_two: %d\n", kappa_two);
                    // }
                }
                // if(world_rank == 1) {
                //     for(int i = 0; i < process_m * k; i++) {
                //         printf("%f\n", temp_dist[i]);
                //     }
                //     printf("\n");
                // }
                for(int kappa = 0; kappa < k; kappa++) {
                    sub_knnresult.ndist[kappa*process_m + i] = temp_dist[kappa];
                    sub_knnresult.nidx[kappa*process_m + i] = temp_index[kappa];
                }             
            }

            // if(world_rank == 1) {
            //     printf("SUB KNN RESULT\n");
            //     for(int i = 0; i < process_m * k; i++) {
            //         printf("%f\n", sub_knnresult.ndist[i]);
            //     }
            //     printf("\n");
            // }
            // Don't forget to clear temp_knnresult after this comment
            free(temp_knnresult.ndist);
            free(temp_knnresult.nidx);

            // if(world_rank == 1) {
            //     for(int i = 0; i < process_m * k; i++) {
            //         printf("%f\n", temp_knnresult.ndist[i]);
            //     }
            // }
            // printf("\n");

            // if(world_rank == 1) {
            //     printf("\n");
            //     for(int i = 0; i < process_m * k; i++) {
            //         printf("%f\n", temp_knnresult.ndist[i]);
            //     }
            //     printf("\n");
            // }
        }                
    }

    // if(world_rank == 1) {
    //     for(int i = 0; i < process_m * k; i++) {
    //         printf("%f\n", sub_knnresult.ndist[i]);
    //     }
    // }

    // Send the sub_knn_results to zero process
    if(world_rank != 0) {
        MPI_Send(sub_knnresult.ndist, process_m * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else {
        knnresult knnResult;
        knnResult.nidx = malloc(n * k * sizeof(int));
        if(knnResult.nidx == NULL) {
            printf("error allocating memory\n");
            exit(0);
        }
        knnResult.ndist = malloc(n * k * sizeof(double));
        if(knnResult.ndist == NULL) {
            printf("error allocating memory\n");
            exit(0);
        }
        knnResult.n = n;
        knnResult.k = k;
        for(int i = 1; i < p; i++) {
            double *temp = malloc(process_m * k * sizeof(double));
            MPI_Recv(temp, process_m * k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Position the elements in the appropriate position of the knnResult.ndist array
            printf("\n");
            for(int j = 0; j < process_m; j++) {
                for(int kappa = 0; kappa < k; kappa++) {
                    printf("%f\n", temp[process_m * kappa + j]);
                    knnResult.ndist[i * process_m + j + kappa*n] = temp[process_m * kappa + j];
                    printf("%d\n", i * process_m + j + kappa*n);
                }
            }
        }
        // For i = 0
        for(int j = 0; j < process_m; j++) {
            for(int kappa = 0; kappa < k; kappa++) {
                printf("%f\n", sub_knnresult.ndist[process_m * kappa + j]);
                knnResult.ndist[j + kappa*n] = sub_knnresult.ndist[process_m * kappa + j];
                printf("%d\n", j + kappa*n);
            }
        }

        printf("FINAL KNN");
        for(int i = 0; i < n * k ; i++) {
            // if(i % n == 0) {
                // printf("\n");
            // }
            printf("%f \n", knnResult.ndist[i]);
        }
    }


    
    // Finalize the MPI environment.
    MPI_Finalize();
}