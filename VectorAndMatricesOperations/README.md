
## 📌 TL;DR

If you work with CUDA, this is a quick sanity check to see how threads, blocks, warps, and shared memory really behave in practice.

---

# 🚀 GPU Reminder — Vector & Matrix Multiplication with CUDA

A small hands-on project to visualize how GPUs actually behave with threads, blocks, warps, and SMs.

This is not meant to be fancy or production-ready. The goal is simple: validate assumptions about CUDA execution and memory behavior by comparing CPU vs GPU on classic workloads.

---

## ✨ What’s Implemented

- **Vector Addition**
  - 16M elements
  - CPU vs GPU comparison

- **Matrix Multiplication**
  - Naive implementation
  - Optimized tiled version using shared memory

- **Shared Logic**
  - Kernels using `__host__ __device__` to reuse logic on CPU and GPU

---

## 📊 Benchmarks

### Vector Addition (16M elements)

| Device | Time |
|--------|------|
| CPU    | ~34 ms |
| GPU    | ~17 ms |

**Speedup:** ~1.9× faster on GPU

---

### Matrix Multiplication (1024 × 1024)

| Version        | CPU Time | GPU Time | Speedup |
|---------------|---------|---------|---------|
| Naive         | ~6.4 s  | 5–7 ms  | 940× – 1300× |
| Optimized     | ~6 s    | ~3.8 ms | ~1560× |

---

## 🧠 Key Takeaways

- Block size has a direct impact on **occupancy**
- Registers and shared memory impose real limits on **parallelism**
- Simple **tiling + shared memory** drastically reduces runtime
- GPU scheduling theory becomes much clearer when validated with numbers

---

## 🎯 Purpose

This project is just a practical reminder of how GPU scheduling and memory hierarchies work in real conditions.

Nothing fancy — just confirming theory with actual measurements.

---

## 🛠 Tech

- CUDA C++
- CPU reference implementations
- Naive and tiled GPU kernels

---

## 🚧 Notes

Results depend on:
- GPU architecture
- Block size
- Memory layout
- Compiler flags

Treat numbers as **indicative**, not absolute.

---
Theory is good. Numbers are better.
