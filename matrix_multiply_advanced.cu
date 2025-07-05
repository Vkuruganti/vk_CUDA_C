#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

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

// Tile size for shared memory optimization
#define TILE_SIZE 32

// Naive matrix multiplication kernel
__global__ void matrixMultiplyNaive(float* A, float* B, float* C, 
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

// Tiled matrix multiplication with shared memory
__global__ void matrixMultiplyTiled(float* A, float* B, float* C, 
                                   int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if (tile * TILE_SIZE + threadIdx.y < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Coalesced memory access version
__global__ void matrixMultiplyCoalesced(float* A, float* B, float* C, 
                                       int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A into shared memory (coalesced access)
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory (coalesced access)
        if (tile * TILE_SIZE + threadIdx.y < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Warp-level optimization version
__global__ void matrixMultiplyWarpOptimized(float* A, float* B, float* C, 
                                           int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if (tile * TILE_SIZE + threadIdx.y < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product with warp-level optimization
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result with warp-level optimization
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU reference implementation
void matrixMultiplyCPU(float* A, float* B, float* C, int M, int N, int K) {
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

// Initialize matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Print matrix (for small matrices)
void printMatrix(float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Verify results
bool verifyResults(float* cpu_result, float* gpu_result, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: CPU=%f, GPU=%f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

// Time a kernel execution
float timeKernel(void (*kernel)(float*, float*, float*, int, int, int),
                float* d_A, float* d_B, float* d_C, int M, int N, int K,
                dim3 gridDim, dim3 blockDim, const char* kernel_name) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm up
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Time the kernel
    CHECK_CUDA(cudaEventRecord(start));
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return time;
}

int main(int argc, char** argv) {
    // Matrix dimensions
    int M = 1024;  // Rows of A
    int N = 1024;  // Columns of B
    int K = 1024;  // Columns of A / Rows of B
    
    // Parse command line arguments
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    
    printf("Advanced Matrix Multiplication: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    
    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_C_CPU = (float*)malloc(sizeC);
    
    if (!h_A || !h_B || !h_C || !h_C_CPU) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }
    
    // Initialize matrices
    srand(42);  // Fixed seed for reproducibility
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);
    
    // Print small matrices for verification
    if (M <= 8 && N <= 8 && K <= 8) {
        printMatrix(h_A, M, K, "Matrix A");
        printMatrix(h_B, K, N, "Matrix B");
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // CPU reference computation
    clock_t cpu_start = clock();
    matrixMultiplyCPU(h_A, h_B, h_C_CPU, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    
    // Test different kernels
    float naive_time = timeKernel(matrixMultiplyNaive, d_A, d_B, d_C, M, N, K, 
                                 gridDim, blockDim, "Naive");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    bool naive_correct = verifyResults(h_C_CPU, h_C, M * N);
    
    float tiled_time = timeKernel(matrixMultiplyTiled, d_A, d_B, d_C, M, N, K, 
                                 gridDim, blockDim, "Tiled");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    bool tiled_correct = verifyResults(h_C_CPU, h_C, M * N);
    
    float coalesced_time = timeKernel(matrixMultiplyCoalesced, d_A, d_B, d_C, M, N, K, 
                                     gridDim, blockDim, "Coalesced");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    bool coalesced_correct = verifyResults(h_C_CPU, h_C, M * N);
    
    float warp_time = timeKernel(matrixMultiplyWarpOptimized, d_A, d_B, d_C, M, N, K, 
                                gridDim, blockDim, "Warp Optimized");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    bool warp_correct = verifyResults(h_C_CPU, h_C, M * N);
    
    // Print results
    printf("\n=== Performance Results ===\n");
    printf("CPU Time: %.3f ms\n", cpu_time);
    printf("GPU Naive Time: %.3f ms (Speedup: %.2fx)\n", 
           naive_time, cpu_time / naive_time);
    printf("GPU Tiled Time: %.3f ms (Speedup: %.2fx)\n", 
           tiled_time, cpu_time / tiled_time);
    printf("GPU Coalesced Time: %.3f ms (Speedup: %.2fx)\n", 
           coalesced_time, cpu_time / coalesced_time);
    printf("GPU Warp Optimized Time: %.3f ms (Speedup: %.2fx)\n", 
           warp_time, cpu_time / warp_time);
    
    printf("\n=== Correctness Results ===\n");
    printf("Naive Kernel Correct: %s\n", naive_correct ? "YES" : "NO");
    printf("Tiled Kernel Correct: %s\n", tiled_correct ? "YES" : "NO");
    printf("Coalesced Kernel Correct: %s\n", coalesced_correct ? "YES" : "NO");
    printf("Warp Optimized Kernel Correct: %s\n", warp_correct ? "YES" : "NO");
    
    // Print small result matrix
    if (M <= 8 && N <= 8) {
        printMatrix(h_C_CPU, M, N, "CPU Result");
        printMatrix(h_C, M, N, "GPU Result (Warp Optimized)");
    }
    
    // Calculate FLOPS
    long long flops = 2LL * M * N * K;
    printf("\n=== FLOPS Analysis ===\n");
    printf("Total FLOPS: %lld\n", flops);
    printf("GPU Naive GFLOPS: %.2f\n", (flops / 1e9) / (naive_time / 1000.0));
    printf("GPU Tiled GFLOPS: %.2f\n", (flops / 1e9) / (tiled_time / 1000.0));
    printf("GPU Coalesced GFLOPS: %.2f\n", (flops / 1e9) / (coalesced_time / 1000.0));
    printf("GPU Warp Optimized GFLOPS: %.2f\n", (flops / 1e9) / (warp_time / 1000.0));
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    return 0;
} 