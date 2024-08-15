#!/bin/bash

# Number of samples for each implementation
SAMPLES=10

# Output file
OUTPUT_FILE="benchmark_results.csv"

# Write header to output file
echo "implementation,duration,gflops" >$OUTPUT_FILE

# Run benchmarks
for i in $(seq 1 $SAMPLES); do
  ./matrix_multiply_dpcpp >>$OUTPUT_FILE
  ./matrix_multiply_standard >>$OUTPUT_FILE
done

echo "Benchmark complete. Results saved to $OUTPUT_FILE"
