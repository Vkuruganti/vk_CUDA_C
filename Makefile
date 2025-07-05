# CUDA Matrix Multiplication Makefile

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_60 -std=c++11

# Target executables
TARGET = matrix_multiply
TARGET_ADVANCED = matrix_multiply_advanced
TARGET_EXAMPLE = example

# Source files
SOURCES = matrix_multiply.cu
SOURCES_ADVANCED = matrix_multiply_advanced.cu
SOURCES_EXAMPLE = example.cu

# Default target
all: $(TARGET) $(TARGET_ADVANCED) $(TARGET_EXAMPLE)

# Compile the basic CUDA program
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

# Compile the advanced CUDA program
$(TARGET_ADVANCED): $(SOURCES_ADVANCED)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET_ADVANCED) $(SOURCES_ADVANCED)

# Compile the example program
$(TARGET_EXAMPLE): $(SOURCES_EXAMPLE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET_EXAMPLE) $(SOURCES_EXAMPLE)

# Debug version
debug: NVCC_FLAGS += -g -G
debug: $(TARGET) $(TARGET_ADVANCED) $(TARGET_EXAMPLE)

# Profile version
profile: NVCC_FLAGS += -pg
profile: $(TARGET) $(TARGET_ADVANCED) $(TARGET_EXAMPLE)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET_ADVANCED) $(TARGET_EXAMPLE) *.o

# Run basic version with default parameters (1024x1024)
run: $(TARGET)
	./$(TARGET)

# Run advanced version with default parameters
run_advanced: $(TARGET_ADVANCED)
	./$(TARGET_ADVANCED)

# Run example
run_example: $(TARGET_EXAMPLE)
	./$(TARGET_EXAMPLE)

# Run with small matrices for verification
test: $(TARGET)
	./$(TARGET) 4 4 4

test_advanced: $(TARGET_ADVANCED)
	./$(TARGET_ADVANCED) 4 4 4

# Run with medium matrices
medium: $(TARGET)
	./$(TARGET) 512 512 512

medium_advanced: $(TARGET_ADVANCED)
	./$(TARGET_ADVANCED) 512 512 512

# Run with large matrices
large: $(TARGET)
	./$(TARGET) 2048 2048 2048

large_advanced: $(TARGET_ADVANCED)
	./$(TARGET_ADVANCED) 2048 2048 2048

# Show GPU information
info:
	nvidia-smi

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build all programs (basic, advanced, example)"
	@echo "  debug            - Build with debug symbols"
	@echo "  profile          - Build with profiling enabled"
	@echo "  clean            - Remove build artifacts"
	@echo ""
	@echo "Basic Matrix Multiplication:"
	@echo "  run              - Run basic version with 1024x1024 matrices"
	@echo "  test             - Run basic version with 4x4 matrices"
	@echo "  medium           - Run basic version with 512x512 matrices"
	@echo "  large            - Run basic version with 2048x2048 matrices"
	@echo ""
	@echo "Advanced Matrix Multiplication:"
	@echo "  run_advanced     - Run advanced version with 1024x1024 matrices"
	@echo "  test_advanced    - Run advanced version with 4x4 matrices"
	@echo "  medium_advanced  - Run advanced version with 512x512 matrices"
	@echo "  large_advanced   - Run advanced version with 2048x2048 matrices"
	@echo ""
	@echo "Example Program:"
	@echo "  run_example      - Run simple example with 4x4 matrices"
	@echo ""
	@echo "  info             - Show GPU information"
	@echo "  help             - Show this help message"

.PHONY: all debug profile clean run run_advanced run_example test test_advanced medium medium_advanced large large_advanced info help 