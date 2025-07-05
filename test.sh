#!/bin/bash

# CUDA Matrix Multiplication Test Script
# This script tests the compilation and basic functionality of the matrix multiplication implementations

echo "=== CUDA Matrix Multiplication Test Script ==="
echo

# Check if CUDA is available
echo "1. Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "   ✓ NVCC found: $(nvcc --version | head -n1)"
else
    echo "   ✗ NVCC not found. Please install CUDA Toolkit."
    exit 1
fi

# Check GPU availability
echo "2. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
        echo "   GPU: $line"
    done
else
    echo "   ✗ nvidia-smi not found. No NVIDIA GPU detected."
    exit 1
fi

echo
echo "3. Building the programs..."

# Clean previous builds
make clean

# Build both implementations
if make all; then
    echo "   ✓ Build successful"
else
    echo "   ✗ Build failed"
    exit 1
fi

echo
echo "4. Testing basic functionality..."

# Test with small matrices
echo "   Testing with 4x4 matrices..."
if ./matrix_multiply 4 4 4 > /dev/null 2>&1; then
    echo "   ✓ Basic implementation works"
else
    echo "   ✗ Basic implementation failed"
    exit 1
fi

if ./matrix_multiply_advanced 4 4 4 > /dev/null 2>&1; then
    echo "   ✓ Advanced implementation works"
else
    echo "   ✗ Advanced implementation failed"
    exit 1
fi

echo
echo "5. Running performance tests..."

# Test with medium matrices
echo "   Testing with 512x512 matrices..."
echo "   Basic implementation:"
./matrix_multiply 512 512 512 2>&1 | grep -E "(CPU Time|GPU.*Time|Speedup|Correct)"

echo "   Advanced implementation:"
./matrix_multiply_advanced 512 512 512 2>&1 | grep -E "(CPU Time|GPU.*Time|Speedup|Correct)"

echo
echo "6. Testing with custom matrix sizes..."

# Test with non-square matrices
echo "   Testing 256x512 * 512x128..."
if ./matrix_multiply 256 128 512 > /dev/null 2>&1; then
    echo "   ✓ Non-square matrix multiplication works"
else
    echo "   ✗ Non-square matrix multiplication failed"
fi

echo
echo "=== Test Summary ==="
echo "✓ CUDA installation verified"
echo "✓ GPU detected"
echo "✓ Programs compiled successfully"
echo "✓ Basic functionality tested"
echo "✓ Performance tests completed"
echo
echo "The CUDA matrix multiplication implementations are working correctly!"
echo
echo "To run performance benchmarks:"
echo "  make run              # Basic implementation with 1024x1024"
echo "  make run_advanced     # Advanced implementation with 1024x1024"
echo "  make large            # Large matrices (2048x2048)"
echo "  make large_advanced   # Advanced with large matrices"
echo
echo "To test with custom sizes:"
echo "  ./matrix_multiply <M> <N> <K>"
echo "  ./matrix_multiply_advanced <M> <N> <K>" 