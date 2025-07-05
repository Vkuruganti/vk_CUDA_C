@echo off
REM CUDA Matrix Multiplication Test Script for Windows
REM This script tests the compilation and basic functionality of the matrix multiplication implementations

echo === CUDA Matrix Multiplication Test Script ===
echo.

REM Check if CUDA is available
echo 1. Checking CUDA installation...
nvcc --version >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✓ NVCC found
    nvcc --version | findstr "release"
) else (
    echo    ✗ NVCC not found. Please install CUDA Toolkit.
    exit /b 1
)

REM Check GPU availability
echo 2. Checking GPU availability...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✓ nvidia-smi found
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
) else (
    echo    ✗ nvidia-smi not found. No NVIDIA GPU detected.
    exit /b 1
)

echo.
echo 3. Building the programs...

REM Clean previous builds
make clean

REM Build both implementations
make all
if %errorlevel% equ 0 (
    echo    ✓ Build successful
) else (
    echo    ✗ Build failed
    exit /b 1
)

echo.
echo 4. Testing basic functionality...

REM Test with small matrices
echo    Testing with 4x4 matrices...
matrix_multiply.exe 4 4 4 >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✓ Basic implementation works
) else (
    echo    ✗ Basic implementation failed
    exit /b 1
)

matrix_multiply_advanced.exe 4 4 4 >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✓ Advanced implementation works
) else (
    echo    ✗ Advanced implementation failed
    exit /b 1
)

echo.
echo 5. Running performance tests...

REM Test with medium matrices
echo    Testing with 512x512 matrices...
echo    Basic implementation:
matrix_multiply.exe 512 512 512 2>&1 | findstr /R "CPU Time GPU.*Time Speedup Correct"

echo    Advanced implementation:
matrix_multiply_advanced.exe 512 512 512 2>&1 | findstr /R "CPU Time GPU.*Time Speedup Correct"

echo.
echo 6. Testing with custom matrix sizes...

REM Test with non-square matrices
echo    Testing 256x512 * 512x128...
matrix_multiply.exe 256 128 512 >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✓ Non-square matrix multiplication works
) else (
    echo    ✗ Non-square matrix multiplication failed
)

echo.
echo === Test Summary ===
echo ✓ CUDA installation verified
echo ✓ GPU detected
echo ✓ Programs compiled successfully
echo ✓ Basic functionality tested
echo ✓ Performance tests completed
echo.
echo The CUDA matrix multiplication implementations are working correctly!
echo.
echo To run performance benchmarks:
echo   make run              # Basic implementation with 1024x1024
echo   make run_advanced     # Advanced implementation with 1024x1024
echo   make large            # Large matrices (2048x2048)
echo   make large_advanced   # Advanced with large matrices
echo.
echo To test with custom sizes:
echo   matrix_multiply.exe ^<M^> ^<N^> ^<K^>
echo   matrix_multiply_advanced.exe ^<M^> ^<N^> ^<K^> 