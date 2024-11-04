# Parallel-And-Distributed-Systems-Exercise-2

## Algorithm V0 analysis

The serial search of the K-Nearest Neighbours required the simple application of the formula given. Through the function cblas_dgemm the multiplication of X, Y was performed and by simple iterations on the contents of X, Y the final matrix D was computed. Then, equally important was the code to search for the k nearest neighbours by searching the matrix D and setting the already selected values equal to infinity. This returned the final `knnresultstruct` with the nearest neighbour and nearest distance tables for each element of Y based on the original matrix X.

## Algorithm V1 analysis

In this version of the algorithm, the calculation of the final knnresult was done simultaneously using several processors, essentially splitting the problem into subproblems. Initially, the matrix of X was known only to the processor 0 and distributed sub-tables of X to each of the other processors.This was accomplished through a for-loop that distributed the appropriate pointer to the original matrix and the range of values, so that each processor would know the elements of the original problem that were assigned to it, via the `MPI_Send` and `MPI_Recv` instructions. Then, it was considered more efficient to share, through a ring structure, the sub-tables between the processors, in order to search for the K-Nearest Neighbours (through the previous function knn of v0) and at the end each processor has calculated its own knn result for its initial sub-table. That means that matrices X were shared during the calculation while matrix Y was always fixed for each process. Once this process was finished, each sub-table was broadcast back to the original process in order to assemble the complete result.

## Algorithm V2 analysis

Following on from the previous process, and in line with the excellent idea of vptrees, the above code could be made even faster by creating a neighbour search tree. Thus, after each processor receives its own subtable X, the vpt_create function is called which creates the vptree corresponding to the specific X elements, which is separate and fixed for each processor. Then, using the same ring structure, the data is swapped (note: this time, in addition to the data, the knnresult tables are also swapped in order to preserve the nearest neighbour information) and then the nearest neighbours are updated via the search_vpt function for the specific subset of its elements, based on that processor's vptree (which is fixed for each node). In the end, each element of the subtable will have been compared to all vptreesand kept the closest distances, saving a lot of unnecessary distance calculations!

## Diagrams

- V0 Diagrams

![alt text](/diagrams/v0.png "Diagram")

From the diagrams of V0 we observe that k is the main factor that proportionally affects the execution times of this implementation, i.e., the number of nearest neighbours searched, rather than the number of dimensions d of the space.

- V1 Diagrams

![alt text](/diagrams/v1/n=20000p=2.jpg "Diagram")
![alt text](/diagrams/v1/n=20000p=4.jpg "Diagram")
![alt text](/diagrams/v1/n=50000p=4.jpg "Diagram")
![alt text](/diagrams/v1/n=32000p=8.jpg "Diagram")
![alt text](/diagrams/v1/n=64000p=8.jpg "Diagram")
![alt text](/diagrams/v1/n=60000p=15.jpg "Diagram")
![alt text](/diagrams/v1/n=90000p=15.jpg "Diagram")

From the diagrams above we deduct that for V1, regardless of the number of processors used in the computation, the largest impact (if not the only one) on the execution time is the factor k, as observed in V0. We also conclude that for an equal number of elements n but for an increased number of processors, there is a decrease in time, but when the opposite occurs, that is, the number of processors is constant but n increases there is an increase in execution time. Finally, when comparing V1 and V0 for n= 20000 we observe that V1 shows much less runtime.

- V2 Diagrams

![alt text](/diagrams/v2/n=20000p=2.jpg "Diagram")
![alt text](/diagrams/v2/n=20000p=4.jpg "Diagram")
![alt text](/diagrams/v2/n=50000p=4.jpg "Diagram")
![alt text](/diagrams/v2/n=64000p=4.jpg "Diagram")
![alt text](/diagrams/v2/n=32000p=8.jpg "Diagram")
![alt text](/diagrams/v2/n=64000p=8.jpg "Diagram")
![alt text](/diagrams/v2/n=60000p=15.jpg "Diagram")
![alt text](/diagrams/v2/n=90000p=15.jpg "Diagram")

From the diagrams above we first see that for the  same experiments as V2 we have much lower runtimes compared to the V1 implementation! We also notice that the effect of the factor k (especially for d>7) starts to become negligible.
