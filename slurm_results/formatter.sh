#!/bin/bash

files=(slurm*)
touch results.txt

for file in ${files[@]}
do
    while read line; do
        if [[ $line == Duration:* ]] ;
        then
            echo  $line | grep -o '[0-9]*\.[0-9]*' | xargs -I '{}' echo  "{}; " >> results.txt
        fi    

        if [[ $line == d:* ]] || [[ $line == k:* ]] || [[ $line == Processes:* ]] || [[ $line == n:* ]];
        then
            echo -n $line | grep -o '[0-9]*' | xargs -I '{}' echo -n "{}; " >> results.txt
        fi   
    done < $file
done
