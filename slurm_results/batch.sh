#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --time=0:30:00

module load gcc openmpi
module load openblas

n=$1
k=$2
d=$3 

srun -n 2 ../out/v1_1 $n $k $d

#srun -n 2 ../out/v1_1 20000 3 10
#srun -n 2 ../out/v1_1 20000 3 25
#srun -n 2 ../out/v1_1 20000 3 50
#srun -n 2 ../out/v1_1 20000 3 75
#srun -n 2 ../out/v1_1 20000 3 100
#
#srun -n 2 ../out/v1_1 20000 7 10
#srun -n 2 ../out/v1_1 20000 7 25
#srun -n 2 ../out/v1_1 20000 7 50
#srun -n 2 ../out/v1_1 20000 7 75
#srun -n 2 ../out/v1_1 20000 7 100
#
#srun -n 2 ../out/v1_1 20000 15 10
#srun -n 2 ../out/v1_1 20000 15 25
#srun -n 2 ../out/v1_1 20000 15 50
#srun -n 2 ../out/v1_1 20000 15 75
#srun -n 2 ../out/v1_1 20000 15 100
#
#srun -n 2 ../out/v1_1 20000 20 10
#srun -n 2 ../out/v1_1 20000 20 25
#srun -n 2 ../out/v1_1 20000 20 50
#srun -n 2 ../out/v1_1 20000 20 75
#srun -n 2 ../out/v1_1 20000 20 100
#
#srun -n 2 ../out/v1_1 40000 3 10
#srun -n 2 ../out/v1_1 40000 3 25
#srun -n 2 ../out/v1_1 40000 3 50
#srun -n 2 ../out/v1_1 40000 3 75
#srun -n 2 ../out/v1_1 40000 3 100
#
#srun -n 2 ../out/v1_1 40000 7 10
#srun -n 2 ../out/v1_1 40000 7 25
#srun -n 2 ../out/v1_1 40000 7 50
#srun -n 2 ../out/v1_1 40000 7 75
#srun -n 2 ../out/v1_1 40000 7 100
#
#srun -n 2 ../out/v1_1 40000 15 10
#srun -n 2 ../out/v1_1 40000 15 25
#srun -n 2 ../out/v1_1 40000 15 50
#srun -n 2 ../out/v1_1 40000 15 75
#srun -n 2 ../out/v1_1 40000 15 100
#
#srun -n 2 ../out/v1_1 40000 20 10
#srun -n 2 ../out/v1_1 40000 20 25
#srun -n 2 ../out/v1_1 40000 20 50
#srun -n 2 ../out/v1_1 40000 20 75
#srun -n 2 ../out/v1_1 40000 20 100