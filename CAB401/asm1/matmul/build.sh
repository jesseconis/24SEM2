#!/bin/bash

. /opt/intel/oneapi/setvars.sh

# Compile the Intel DPC++ version
icpx -fsycl -O3 -std=c++17 -o matrix_multiply_dpcpp matrix_multiply_dpcpp.cpp

# Compile the standard C++ version
g++ -O3 -std=c++17 -o matrix_multiply_standard matrix_multiply_standard.cpp

echo "Compilation complete. Run the executables with:"
echo "./matrix_multiply_dpcpp"
echo "./matrix_multiply_standard"
