# Introduction

A simple example for CUDA gemm with nccl impl, using 2 gpus.

$C = AB + C$,

# How to build and test

```bash
# build
make

# run
./main

# clean
make clean

```

# Environment

- python3: build nccl
- nccl: install the nccl to cuda path
- cuda-tool-kit: need cublas to verify results 
