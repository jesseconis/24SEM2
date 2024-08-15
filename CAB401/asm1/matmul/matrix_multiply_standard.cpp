#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Function to initialize a matrix with random values
void initialize_matrix(std::vector<float>& matrix, int rows, int cols, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

// Function to perform matrix multiplication using standard C++
void matrix_multiply_standard(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                              int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
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
    matrix_multiply_standard(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double gflops = (2.0 * M * N * K) / (duration.count() * 1e9);

    // Output in CSV-like format: implementation,duration,gflops
    std::cout << "Standard," << duration.count() << "," << gflops << std::endl;

    return 0;
}
