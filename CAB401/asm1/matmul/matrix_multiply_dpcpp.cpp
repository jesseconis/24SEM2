#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace sycl;

// Function to initialize a matrix with random values
void initialize_matrix(std::vector<float>& matrix, int rows, int cols, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

// Function to perform matrix multiplication using Intel DPC++
void matrix_multiply_dpcpp(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                           int M, int N, int K) {
    queue q;

    buffer<float, 2> buf_A(A.data(), range<2>(M, K));
    buffer<float, 2> buf_B(B.data(), range<2>(K, N));
    buffer<float, 2> buf_C(C.data(), range<2>(M, N));

    auto event = q.submit([&](handler& h) {
        auto a = buf_A.get_access<access::mode::read>(h);
        auto b = buf_B.get_access<access::mode::read>(h);
        auto c = buf_C.get_access<access::mode::write>(h);

        h.parallel_for(range<2>(M, N), [=](id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += a[row][i] * b[i][col];
            }
            c[row][col] = sum;
        });
    });

    event.wait();
}

int main() {
    const int M = 1000;  // Number of rows in A and C
    const int N = 1000;  // Number of columns in B and C
    const int K = 1000;  // Number of columns in A and rows in B
    const unsigned int SEED = 12345;  // Fixed seed for reproducibility

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    initialize_matrix(A, M, K, SEED);
    initialize_matrix(B, K, N, SEED + 1);

    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply_dpcpp(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double gflops = (2.0 * M * N * K) / (duration.count() * 1e9);

    // Output in CSV-like format: implementation,duration,gflops
    std::cout << "DPC++," << duration.count() << "," << gflops << std::endl;

    return 0;
}




























