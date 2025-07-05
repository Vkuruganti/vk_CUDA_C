# CUDA Matrix Multiplication

This project implements high-performance matrix multiplication using NVIDIA CUDA, demonstrating various optimization techniques for GPU computing.

## Overview

Matrix multiplication is a fundamental operation in scientific computing, machine learning, and image processing. This project provides:

1. **Basic Implementation** (`matrix_multiply.cu`): Simple CUDA kernels with performance comparison
2. **Advanced Implementation** (`matrix_multiply_advanced.cu`): Multiple optimization techniques
3. **Comprehensive Testing**: CPU vs GPU performance comparison and correctness verification

## Features

### Basic Implementation
- Naive CUDA kernel
- Optimized kernel with shared memory
- CPU reference implementation
- Performance benchmarking
- Correctness verification

### Advanced Implementation
- **Naive Kernel**: Basic parallel implementation
- **Tiled Kernel**: Shared memory optimization
- **Coalesced Kernel**: Memory access optimization
- **Warp Optimized Kernel**: Loop unrolling and warp-level optimizations
- Comprehensive performance analysis
- Multiple matrix size testing

## Optimization Techniques

### 1. Shared Memory Tiling
- Divides matrices into tiles that fit in shared memory
- Reduces global memory accesses
- Improves memory bandwidth utilization

### 2. Memory Coalescing
- Ensures threads in a warp access consecutive memory locations
- Maximizes memory bandwidth
- Critical for performance on modern GPUs

### 3. Loop Unrolling
- Reduces loop overhead
- Improves instruction-level parallelism
- Better register utilization

### 4. Warp-Level Optimizations
- Exploits warp execution characteristics
- Reduces thread divergence
- Optimizes synchronization

## Building and Running

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 11.0 or later recommended)
- GCC or compatible C++ compiler

### Compilation
```bash
# Build both implementations
make all

# Build with debug symbols
make debug

# Build with profiling enabled
make profile

# Clean build artifacts
make clean
```

### Running the Programs

#### Basic Implementation
```bash
# Run with default 1024x1024 matrices
make run

# Test with small matrices (4x4)
make test

# Run with medium matrices (512x512)
make medium

# Run with large matrices (2048x2048)
make large
```

#### Advanced Implementation
```bash
# Run with default 1024x1024 matrices
make run_advanced

# Test with small matrices (4x4)
make test_advanced

# Run with medium matrices (512x512)
make medium_advanced

# Run with large matrices (2048x2048)
make large_advanced
```

#### Custom Matrix Sizes
```bash
# Basic implementation: A(MxK) * B(KxN) = C(MxN)
./matrix_multiply <M> <N> <K>

# Advanced implementation: A(MxK) * B(KxN) = C(MxN)
./matrix_multiply_advanced <M> <N> <K>

# Examples:
./matrix_multiply 1024 1024 1024
./matrix_multiply_advanced 2048 2048 2048
```

## Performance Analysis

The programs provide comprehensive performance metrics:

- **Execution Time**: CPU vs GPU comparison
- **Speedup**: GPU performance improvement over CPU
- **GFLOPS**: Floating-point operations per second
- **Correctness Verification**: Ensures GPU results match CPU reference

### Expected Performance

On modern GPUs (RTX 3080, RTX 4080, etc.):
- **Naive Kernel**: 100-500 GFLOPS
- **Tiled Kernel**: 500-1000 GFLOPS
- **Optimized Kernels**: 1000-2000+ GFLOPS

Speedup over CPU typically ranges from 10x to 100x depending on matrix size and GPU.

## Code Structure

### Basic Implementation (`matrix_multiply.cu`)
```c
// Naive kernel - direct parallel implementation
__global__ void matrixMultiplyNaive(float* A, float* B, float* C, int M, int N, int K)

// Optimized kernel - shared memory tiling
__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int M, int N, int K)
```

### Advanced Implementation (`matrix_multiply_advanced.cu`)
```c
// Multiple kernel variants with different optimizations
__global__ void matrixMultiplyNaive(...)
__global__ void matrixMultiplyTiled(...)
__global__ void matrixMultiplyCoalesced(...)
__global__ void matrixMultiplyWarpOptimized(...)
```

## Key Concepts

### CUDA Memory Hierarchy
- **Global Memory**: Large, slow memory accessible by all threads
- **Shared Memory**: Fast, small memory shared within a block
- **Registers**: Fastest memory, private to each thread

### Thread Organization
- **Grid**: Collection of blocks
- **Block**: Collection of threads (up to 1024 threads)
- **Warp**: Group of 32 threads that execute in lockstep

### Memory Access Patterns
- **Coalesced Access**: Threads access consecutive memory locations
- **Strided Access**: Threads access memory with gaps
- **Bank Conflicts**: Multiple threads access same shared memory bank

## Troubleshooting

### Common Issues

1. **CUDA Driver Version Mismatch**
   ```bash
   # Check CUDA version
   nvcc --version
   nvidia-smi
   ```

2. **Memory Allocation Errors**
   - Reduce matrix size
   - Check available GPU memory: `nvidia-smi`

3. **Compilation Errors**
   - Ensure CUDA Toolkit is properly installed
   - Check GPU architecture compatibility

### Debugging
```bash
# Build with debug symbols
make debug

# Use CUDA-GDB for debugging
cuda-gdb ./matrix_multiply

# Profile with NVIDIA Visual Profiler
nvprof ./matrix_multiply
```

## Performance Tips

1. **Matrix Size**: Use multiples of 32 for optimal performance
2. **Memory Alignment**: Ensure proper memory alignment
3. **Block Size**: 32x32 blocks work well for most cases
4. **Shared Memory**: Use shared memory for frequently accessed data
5. **Memory Coalescing**: Ensure threads access consecutive memory

## Future Enhancements

- Support for different data types (double, half precision)
- Multi-GPU implementation
- Integration with cuBLAS for comparison
- Support for sparse matrices
- Batch matrix multiplication
- Custom matrix layouts (blocked, etc.)

## License

This project is provided as educational material for learning CUDA programming and GPU optimization techniques.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the implementations or add new optimization techniques.
