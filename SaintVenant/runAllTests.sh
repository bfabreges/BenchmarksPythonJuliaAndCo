#!/bin/bash
#
#  This script is supposed to run *all* the test, and then compute
#  the final "report" in Results/
#  Not sure it works everywhere. If it does not, improve it, or enter each
#  directory and look at README.md to know what to do. 

NUM_THREADS=32

for i in C++; do
    echo "Test: " $i
    echo "--- "
    (cd $i; ./compile.sh; ./main1d)
done

for i in Ju; do
    echo  "Test: " $i 
    echo "--- "
    (cd $i; ./run.sh)
done

for i in Python Python-Numba; do
    echo  "Test: " $i
    echo "--- "
    (cd $i; python3 ./main1d.py)
done


for i in C++-OpenMP; do
    echo "Test: " $i
    echo "--- "
    (cd $i; ./compile.sh; OMP_NUM_THREADS=$NUM_THREADS ./main1d)
done

for i in Ju-Threads; do
    echo  "Test: " $i 
    echo "--- "
    (cd $i; JULIA_NUM_THREADS=$NUM_THREADS ./run.sh)
done

for i in Python-Numba-Threads; do
    echo  "Test: " $i
    echo "--- "
    (cd $i; NUMBA_NUM_THREADS=$NUM_THREADS python3 ./main1d.py)
done


echo "Checking for a NVIDIA GPU device"
nvidia-smi --query-gpu=name --format=csv,noheader
if [ $? -eq 0 ]; then
    echo "Test: C++-CUDA"
    echo "--- "
    (cd C++-CUDA; ./compile.sh; ./main1d)

    echo "Test: Ju-CUDA"
    echo "--- "
    (cd Ju-CUDA; ./run.sh)

    echo "Test: Python-Numba-CUDA"
    echo "--- "
    (cd Python-Numba-CUDA; python3 ./main1d.py)
else
    echo "No GPU device found"
fi 


echo " "
echo "Make the report:"
(cd Results; ./Look.py)
echo " "
echo "To replay the report, cd Results/ and run ./Look.py "
echo " "

