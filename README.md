# GPU Compute Projects (CUDA / OpenCL)

This repository contains GPU-accelerated projects focused on understanding and benchmarking parallel computing using **CUDA and OpenCL**.
Each project is self-contained and comes with its own build system.
The goal is mainly experimental: validating assumptions about GPU architecture, scheduling, memory, and performance through practical code.

---

## Current Project

- [`VectorAndMatricesOperations/`](https://github.com/Noctiluce/gpu_stuff/tree/main/VectorAndMatricesOperations) – Vector and matrix operations to explore GPU behavior, occupancy, and memory optimization.

---

## Build

Each project has its own CMake configuration.

Example:

```bash
bash
cd gpu_reminder
mkdir build && cd build
cmake ..
make
```
