#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple matrix multiplication kernel
__global__ void matrixMultiply(float* A, float* B, float* C, 
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Initialize matrix with specific pattern
void initializeMatrix(float* matrix, int rows, int cols, int pattern) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            switch (pattern) {
                case 0: // Identity matrix
                    matrix[i * cols + j] = (i == j) ? 1.0f : 0.0f;
                    break;
                case 1: // Constant matrix
                    matrix[i * cols + j] = 2.0f;
                    break;
                case 2: // Sequential values
                    matrix[i * cols + j] = (float)(i * cols + j + 1);
                    break;
                default: // Random values
                    matrix[i * cols + j] = (float)rand() / RAND_MAX;
                    break;
            }
        }
    }
}

// Print matrix
void printMatrix(float* matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("=== CUDA Matrix Multiplication Example ===\n\n");
    
    // Matrix dimensions
    int M = 4;  // Rows of A
    int N = 4;  // Columns of B
    int K = 4;  // Columns of A / Rows of B
    
    printf("Computing: A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);
    
    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }
    
    // Initialize matrices with different patterns
    initializeMatrix(h_A, M, K, 2);  // Sequential values
    initializeMatrix(h_B, K, N, 1);  // Constant matrix (all 2.0)
    
    // Print input matrices
    printMatrix(h_A, M, K, "Matrix A");
    printMatrix(h_B, K, N, "Matrix B");
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    dim3 blockDim(2, 2);  // 2x2 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    printf("Launching kernel with grid(%d,%d) blocks(%d,%d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Launch kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // Print result
    printMatrix(h_C, M, N, "Result Matrix C");
    
    // Verify result manually (for this specific case)
    printf("Verification:\n");
    printf("Since B is all 2.0, C[i,j] should be 2.0 * sum of row i in A\n");
    for (int i = 0; i < M; i++) {
        float expected = 0.0f;
        for (int k = 0; k < K; k++) {
            expected += h_A[i * K + k] * 2.0f;
        }
        printf("Row %d: Expected %.2f, Got %.2f %s\n", 
               i, expected, h_C[i * N], 
               (fabs(expected - h_C[i * N]) < 1e-5) ? "✓" : "✗");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    printf("\nExample completed successfully!\n");
    return 0;
} 