# Makefile for compiling a CUDA project with NCCL and cuBLAS support

# Compiler and flags
CXX = g++
NVCC = nvcc
NVCCFLAGS = -lcublas -lnccl -std=c++11

CUDA_PATH = /usr/local/cuda
INCLUDE_DIRS = -I$(CUDA_PATH)/include
LIB_DIRS = -L$(CUDA_PATH)/lib64

# Target
TARGET = main

# Source and object files
SRC_DIR = src
CU_SRC = $(SRC_DIR)/gemm.cu
CPP_SRC = $(SRC_DIR)/main.cc
CU_OBJ = $(CU_SRC:.cu=.o)
CPP_OBJ = $(CPP_SRC:.cc=.o)

# Default target
all: $(TARGET)

# Link object files into final executable
$(TARGET): $(CPP_OBJ) $(CU_OBJ)
	$(NVCC) $(CPP_OBJ) $(CU_OBJ) -o $(TARGET) $(NVCCFLAGS) $(INCLUDE_DIRS) $(LIB_DIRS)

# Compile .cu files
$(CU_OBJ): $(CU_SRC)
	$(NVCC) -c $< -o $@ $(INCLUDE_DIRS)

# Compile .cc files
$(CPP_OBJ): $(CPP_SRC)
	$(CXX) -c $< -o $@ $(INCLUDE_DIRS)

# Clean up build files
clean:
	rm -f $(CPP_OBJ) $(CU_OBJ) $(TARGET)
