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

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      n;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

typedef struct minArray {
    int* nidx;
    double* ndist;
} minArray;

typedef struct node {
    double* data;
    double* up;
    int up_index;
    double mu;
    int left_size;
    int right_size;
    struct node* left;
    struct node* right;
} node;

// struct node* newNode(double up) {

//     // Allocate memory for new node
//     node* node = malloc(sizeof(node));

//     // Initialize left and right children as NULL
//     node->left = NULL;
//     node->right = NULL;

//     return node;
// }

node* vpt_create(double* x_i_data, node* root, int m, int d) {
    
    if(m == 0) {
        // root = malloc(sizeof(node));
        // root->up = malloc(d * sizeof(double));
        // for(int i = 0; i < d; i++) {
        //     root->up[i] = x_i_data[i];
        // }
        return NULL;
    }

    // Choosing the root node
    root = malloc(sizeof(node));
    root->up = malloc(d * sizeof(double));
    for(int i = 0; i < d; i++) {
        root->up[i] = x_i_data[i];
    }

    // Allocate space for left and right data
    int left_size = 0;
    int right_size = 0;
    root->left = malloc(sizeof(node));
    root->right = malloc(sizeof(node));

    root->left->data = malloc(d * left_size * sizeof(double));
    root->right->data = malloc(d * right_size * sizeof(double));

    // Calculating the mu
    double sum = 0;
    for(int i = 1; i < m; i++) {
        double temp_sum = 0;
        for(int j = 0; j < d; j++){
            temp_sum += (x_i_data[j] - x_i_data[i*d + j]) * (x_i_data[j] - x_i_data[i*d + j]);
        }
        temp_sum = sqrt(abs(temp_sum));
        // printf("%f\n\n", temp_sum);
        sum += temp_sum;
    }
    root->mu = sum / (m - 1);
    // printf("%f\n", root->mu);

    double value = 0;
    for(int i = 1; i < m; i++) {
        double temp_sum = 0;
        for(int j = 0; j < d; j++){
            // We could store those beforehand
            temp_sum += (x_i_data[j] - x_i_data[i*d + j]) * (x_i_data[j] - x_i_data[i*d + j]);
        }
        temp_sum = sqrt(abs(temp_sum));
        // printf("temp_sum: %f\tsum: %f \n\n", temp_sum, root->mu);
        if(temp_sum < root->mu) {
            left_size++;
            root->left_size = left_size;
            root->left->data = realloc(root->left->data, d * left_size * sizeof(double));
            for(int j = 0; j < d; j++) {
                root->left->data[(left_size-1) * d + j] = x_i_data[i*d + j];
            }
        }
        else {
            right_size++;
            root->right_size = right_size;
            root->right->data = realloc(root->right->data, d * right_size * sizeof(double));
            for(int j = 0; j < d; j++) {
                root->right->data[(right_size-1) * d + j] = x_i_data[i*d + j];
            }
        }
    }

    // printf("\nRoot\n");
    // for(int j = 0; j < d; j++) {
    //     printf("%f\n", root->up[j]);
    // }
    // printf("\nLeft Child\n");
    // for(int i = 0; i < left_size; i++) {
    //     for(int j = 0; j < d; j++) {
    //         printf("%f\n", root->left->data[i*d+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\nRight Child\n");
    // for(int i = 0; i < right_size; i++) {
    //     for(int j = 0; j < d; j++) {
    //         printf("%f\n", root->right->data[i*d+j]);
    //     }
    //     printf("\n");
    // }

    root->left = vpt_create(root->left->data, root->left, left_size, d);
    root->right = vpt_create(root->right->data, root->right, right_size, d);
    return root;
}

void search_vpt(double* x_query, node* root, int d, int k,  minArray* min_array) {

    if(root == NULL) {
        // This means that the previous root was a leaf so we have nowhere else to go to
        return;
    }

    // Calculate the distance from the root vantage point
    double distance = 0;
    for(int j = 0; j < d; j++) {
        distance += root->up[j] * root->up[j] - x_query[j] * x_query[j];
    }
    distance = sqrt(abs(distance));

    for(int i = 0; i < k; i++) {
        if(distance < min_array->ndist[i]) {
            double temp_dist;
            double temp_ind;
            for(int j = i; j < k-1; j++) {
                temp_dist = min_array->ndist[j];
                temp_ind = min_array->nidx[j];
                min_array->ndist[j] = distance;
                min_array->nidx[j] = root->up_index;
                min_array->ndist[j+1] = temp_dist;
                min_array->nidx[j+1] = temp_ind;
            }
        }
    }

    // Find the radius
    double radius = INFINITY;
    for(int i = k - 1; i > -1; i++) {
        if(min_array->ndist[i] != INFINITY) {
            radius = min_array->ndist[i];
            break;
        }
    }

    // Compare distance to mu
    if(distance < root->mu - radius) {
        // Search the left child
        search_vpt(x_query, root->left, d, k, min_array);
    }
    else if(distance > root->mu + radius) {
        // Search the right child
        search_vpt(x_query, root->right, d, k, min_array);
    }
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
        // Initialize variables for zero process
        process_n = chunks;
        process_m = chunks;
        x_i_data = malloc(process_m * d * sizeof(double));
        x_i_data = x_data;

        sub_knnresult.k = k;
        sub_knnresult.n = process_m;
        sub_knnresult.ndist = malloc(process_m * k * sizeof(double));
        sub_knnresult.nidx = malloc(process_m * k * sizeof(int));

        // // Initialize sub_knnresult
        // for(int i = 0; i < process_m * k; i ++) {
        //     sub_knnresult.ndist = INFINITY;
        //     sub_knnresult.nidx = -1;
        // }

        // We can now create the vpt tree of the elements of process zero
        node* root;
        root = malloc(sizeof(node));
        root = vpt_create(x_i_data, root, process_m, d);

        node* current_node = root;

        minArray* min_array = malloc(sizeof(minArray));
        min_array->ndist = malloc(k * sizeof(double));
        min_array->nidx = malloc(k * sizeof(int));

        

        // Prepare the x_query
        double* x_query = malloc(d * sizeof(float));
        for(int i = 0; i < 1; i++) {
            // Initialize min_array
            for(int j = 0; j < k; j++) {
                min_array->ndist[j] = INFINITY;
                min_array->nidx[j] = -1;
            }
            for(int j = 0; j < d; j++) {
                x_query[j] = x_i_data[i * d + j];
            }
            search_vpt(x_query, root, d, k, min_array);
            printf("%f \n",min_array->ndist[1]);
        }   
    }
    else if (world_rank == p - 1)
    {
        //Initialize variables for this process
        process_n = (chunks + n % d);
        process_m = (chunks + n % d);
        x_i_data = malloc(process_m * d * sizeof(double));

        // First receive from the mother process
        MPI_Recv(x_i_data, process_m * d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        sub_knnresult.k = k;
        sub_knnresult.n = process_m;
        sub_knnresult.ndist = malloc(process_m * k * sizeof(double));
        sub_knnresult.nidx = malloc(process_m * k * sizeof(int));   
    }
    else
    {
        //Initialize variables for this process
        process_n = chunks;
        process_m = chunks;
        x_i_data = malloc(process_m * d * sizeof(double));

        // First receive from the mother process
        MPI_Recv(x_i_data, process_m * d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        sub_knnresult.k = k;
        sub_knnresult.n = process_m;
        sub_knnresult.ndist = malloc(process_m * k * sizeof(double));
        sub_knnresult.nidx = malloc(process_m * k * sizeof(int));
    }

    // Every process has processed its own x_i_data
    // Every process has to create its own vp tree first
    // And then we have to find the nearest neighbours for every x_i_data according to this vp tree
    // Now they have to pass around data to each other in a ring mode
    // That has to repeat until each process has processed all possible data aka p times

    // // For the first iteration set the y_i_send_data array that will be passed
    // process_n = chunks;
    // y_i_send_data = malloc(process_n * d * sizeof(double));
    // //REALLY IMPORTANT NOT TO SET THE POINTERS EQUAL HERE
    // memcpy(y_i_send_data, x_i_data, process_m * d * sizeof(double));

    // for(int i = 0; i < p; i ++) {
    //     // Process zero sends first then waits for the last process
    //     if(world_rank != 0) {
    //         process_n = chunks;
    //         y_i_receive_data = malloc(process_n * d * sizeof(double));
    //         MPI_Recv(y_i_receive_data, process_n * d, MPI_DOUBLE, world_rank - 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     }


        
    //     MPI_Send(y_i_send_data, process_n * d, MPI_DOUBLE, (world_rank + 1) % p, i, MPI_COMM_WORLD);
    //     if(world_rank != 0) {
    //         // After sending the previous y data that the process holds, in the next iteration, we want to send the new data we just received
    //         memcpy(y_i_send_data, y_i_receive_data, process_m * d * sizeof(double));
    //     }
    //     if(world_rank == 0) {
    //         process_m = chunks;
    //         y_i_receive_data = malloc(process_n * d * sizeof(double));
    //         MPI_Recv(y_i_receive_data, process_n * d, MPI_DOUBLE, p - 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         // After sending the previous y data that the process holds, in the next iteration, we want to send the new data we just received
    //         memcpy(y_i_send_data, y_i_receive_data, process_m * d * sizeof(double));
    //     }

    //     knnresult temp_knnresult;
    //     temp_knnresult.ndist = malloc(process_m * k * sizeof(double));
    //     temp_knnresult.nidx = malloc(process_m * k * sizeof(int));
        
    //     if(!knn_counter) {
    //         // First call of knn so we store the result in the permanent sub_knnresult
    //         sub_knnresult = kNN(y_i_receive_data, x_i_data, process_n, process_m, d, k, world_rank);
    //         knn_counter++;
    //     }
    //     else {
    //         temp_knnresult = kNN(y_i_receive_data, x_i_data, process_n, process_m, d, k, world_rank);
    //         // We have the sub_knnresult from the previous comparisons and the temp_knnresult from the last knn operation
    //         // We have to store in the sub_knnresult the smallest values from both
    //         for(int i = 0; i < process_m; i++) {
    //             double temp_dist[k];
    //             int temp_index[k];    
    //             int kappa_one = 0;
    //             int kappa_two = 0;        
    //             for(int kappa = 0; kappa < k; kappa++) {
    //                 if(temp_knnresult.ndist[kappa_one*process_m + i] < sub_knnresult.ndist[kappa_two*process_m + i]) {
    //                     temp_dist[kappa] = temp_knnresult.ndist[kappa_one*process_m + i];
    //                     temp_index[kappa] = temp_knnresult.nidx[kappa_one*process_m + i];
    //                     kappa_one++;
    //                 }
    //                 else
    //                 {
    //                     temp_dist[kappa] =  sub_knnresult.ndist[kappa_two*process_m + i];
    //                     temp_index[kappa] =  sub_knnresult.nidx[kappa_two*process_m + i];
    //                     kappa_two++;
    //                 }
    //             }
    //             for(int kappa = 0; kappa < k; kappa++) {
    //                 sub_knnresult.ndist[kappa*process_m + i] = temp_dist[kappa];
    //                 sub_knnresult.nidx[kappa*process_m + i] = temp_index[kappa];
    //             }             
    //         }

    //         // Don't forget to clear temp_knnresult after this comment
    //         free(temp_knnresult.ndist);
    //         free(temp_knnresult.nidx);
    //     }                
    // }

    // // Send the sub_knn_results to zero process
    // if(world_rank != 0) {
    //     MPI_Send(sub_knnresult.ndist, process_m * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    // }
    // else {
    //     knnresult knnResult;
    //     knnResult.nidx = malloc(n * k * sizeof(int));
    //     if(knnResult.nidx == NULL) {
    //         printf("error allocating memory\n");
    //         exit(0);
    //     }
    //     knnResult.ndist = malloc(n * k * sizeof(double));
    //     if(knnResult.ndist == NULL) {
    //         printf("error allocating memory\n");
    //         exit(0);
    //     }
    //     knnResult.n = n;
    //     knnResult.k = k;
    //     for(int i = 1; i < p; i++) {
    //         double *temp = malloc(process_m * k * sizeof(double));
    //         MPI_Recv(temp, process_m * k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
    //         // Position the elements in the appropriate position of the knnResult.ndist array
    //         for(int j = 0; j < process_m; j++) {
    //             for(int kappa = 0; kappa < k; kappa++) {
    //                 knnResult.ndist[i * process_m + j + kappa*n] = temp[process_m * kappa + j];
    //             }
    //         }
    //     }
    //     // For i = 0
    //     for(int j = 0; j < process_m; j++) {
    //         for(int kappa = 0; kappa < k; kappa++) {
    //             knnResult.ndist[j + kappa*n] = sub_knnresult.ndist[process_m * kappa + j];
    //         }
    //     }
    // }

    // Finalize the MPI environment.
    MPI_Finalize();
}