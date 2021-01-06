#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=0:10:00

d=(3 7 15 20)
k=(10 25 50 75 100)
n=(45000 110000)

for i in ${n[@]}; do
  for j in ${k[@]}; do
    for k in ${d[@]}; do
      sbatch batch15.sh $i $j $k
    done
  done
done