#!/bin/bash

EIGEN_ROOT=$EIGEN3_INCLUDE_DIR

echo $CXX
echo $EIGEN_ROOT

$CXX -std=c++14 -march=native -mtune=native -O3 -DNDEBUG  main1d.cpp -o main1d -I$EIGEN_ROOT -fopenmp
