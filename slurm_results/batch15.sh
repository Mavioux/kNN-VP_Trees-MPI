#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=15
#SBATCH --nodes=1
#SBATCH --time=0:30:00

module load gcc openmpi
module load openblas

n=$1
k=$2
d=$3 

srun -n 15 ../out/v1_1 $n $k $d