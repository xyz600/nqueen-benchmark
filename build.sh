#!/bin/bash

/usr/bin/g++ -std=c++17 -Ofast -march=native -fopenmp -o cpu_solver cpu_solver.cpp

nvcc -std=c++14 -O2 --generate-code arch=compute_61,code=sm_61 -g -Xptxas=-v -Xcompiler=-fopenmp -o gpu_solver gpu_solver.cu
