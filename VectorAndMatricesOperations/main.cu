#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

__host__ __device__ float vectorMul(float x, float y) {
    return x * y;
}

__global__ void vectorMulKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = vectorMul(a[i], b[i]);
}

void vectorMulCPU(const std::vector<float>& a,
                  const std::vector<float>& b,
                  std::vector<float>& c)
{
    for (size_t i = 0; i < a.size(); ++i)
        c[i] = vectorMul(a[i], b[i]);
}

// Matrix multiplication, CPU version (it just works)
void matrixMulCPU(const std::vector<float>& A,
                  const std::vector<float>& B,
                  std::vector<float>& C,
                  int M, int N, int K)
{
    // A: M×K, B: K×N, C: M×N
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Simple GPU kernel, one thread per output element
__global__ void matrixMulKernelNaive(const float* A, const float* B, float* C,
                                     int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled version with shared memory (perf++)
__global__ void matrixMulKernelTiled(const float* A, const float* B, float* C,
                                     int M, int N, int K)
{
    const int TILE_SIZE = 32;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // A tiles
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // B tiles
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // tiles mult
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

struct BenchmarkResult {
    double cpuTime;
    double gpuTime;
    double speedup;
    double occupancy;
    bool valid;
};

double calculateOccupancy(const void* kernel, int blockSize, size_t dynamicSMem = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        kernel,
        blockSize,
        dynamicSMem
    );

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int activeThreads = maxActiveBlocks * blockSize;

    return (double)activeThreads / (double)maxThreadsPerSM;
}

template<typename Func>
double measureCPU(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(90, '=') << "\n";
}

void printResult(int blockSize, const BenchmarkResult& res, int iterations) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Block: " << std::setw(4) << blockSize
              << " | CPU: " << std::setw(8) << res.cpuTime
              << " ms | GPU: " << std::setw(8) << res.gpuTime
              << " ms | Speedup: " << std::setw(6) << res.speedup << "x"
              << " | Occupancy: " << std::setw(5) << (res.occupancy * 100) << "%";
    if (!res.valid) std::cout << " [VALIDATION FAILED]";
    std::cout << "\n";
}

void printGPUInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "  GPU INFORMATION\n";
    std::cout << std::string(90, '=') << "\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Number of SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Warp size: " << prop.warpSize << "\n";
    std::cout << std::string(90, '=') << "\n";
}

BenchmarkResult benchmarkVector(int blockSize, int iterations, int N) {
    size_t bytes = N * sizeof(float);
    std::vector<float> a(N), b(N), c_cpu(N), c_gpu(N);

    for (int i = 0; i < N; ++i) {
        a[i] = i * 0.565468584f;
        b[i] = i * 2.684864531f;
    }

    double totalCPU = 0.0, totalGPU = 0.0;
    bool allValid = true;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    double occupancy = calculateOccupancy((const void*)vectorMulKernel, blockSize);

    for (int iter = 0; iter < iterations; ++iter) {
        // Run on CPU first
        double cpuTime = measureCPU([&]() {
            vectorMulCPU(a, b, c_cpu);
        });

        cudaEventRecord(start);
        cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

        int numBlocks = (N + blockSize - 1) / blockSize;
        vectorMulKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

        cudaMemcpy(c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float gpuTime = 0;
        cudaEventElapsedTime(&gpuTime, start, end);

        // Check if results match
        bool valid = true;
        for (int i = 0; i < N; ++i) {
            if (fabs(c_cpu[i] - c_gpu[i]) > 1e-3) {
                valid = false;
                allValid = false;
                break;
            }
        }

        if (valid) {
            totalCPU += cpuTime;
            totalGPU += gpuTime;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    double avgCPU = totalCPU / iterations;
    double avgGPU = totalGPU / iterations;

    return {avgCPU, avgGPU, avgCPU / avgGPU, occupancy, allValid};
}

BenchmarkResult benchmarkMatrix(int blockSize, int iterations, int M, int N, int K, bool useTiled) {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    std::vector<float> A(M * K), B(K * N), C_cpu(M * N), C_gpu(M * N);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) A[i] = (i % 100) * 0.01f;
    for (int i = 0; i < K * N; ++i) B[i] = (i % 100) * 0.01f;

    double totalCPU = 0.0, totalGPU = 0.0;
    bool allValid = true;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int totalThreads = blockSize * blockSize;
    size_t sharedMemSize = useTiled ? 2 * 32 * 32 * sizeof(float) : 0;

    double occupancy;
    if (useTiled) {
        occupancy = calculateOccupancy((const void*)matrixMulKernelTiled, totalThreads, sharedMemSize);
    } else {
        occupancy = calculateOccupancy((const void*)matrixMulKernelNaive, totalThreads);
    }

    for (int iter = 0; iter < iterations; ++iter) {
        double cpuTime = measureCPU([&]() {
            matrixMulCPU(A, B, C_cpu, M, N, K);
        });

        cudaEventRecord(start);
        cudaMemcpy(d_A, A.data(), sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), sizeB, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(blockSize, blockSize);
        dim3 numBlocks((N + blockSize - 1) / blockSize,
                       (M + blockSize - 1) / blockSize);

        if (useTiled) {
            matrixMulKernelTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
        } else {
            matrixMulKernelNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
        }

        cudaMemcpy(C_gpu.data(), d_C, sizeC, cudaMemcpyDeviceToHost);
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float gpuTime = 0;
        cudaEventElapsedTime(&gpuTime, start, end);

        bool valid = true;
        for (int i = 0; i < M * N; ++i) {
            if (fabs(C_cpu[i] - C_gpu[i]) > 1e-2) {
                valid = false;
                allValid = false;
                break;
            }
        }

        if (valid) {
            totalCPU += cpuTime;
            totalGPU += gpuTime;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    double avgCPU = totalCPU / iterations;
    double avgGPU = totalGPU / iterations;

    return {avgCPU, avgGPU, avgCPU / avgGPU, occupancy, allValid};
}

int main() {
    const int iterations = 10;

    printGPUInfo();

    printHeader("VECTOR MULTIPLICATION");
    std::cout << "Size: " << (1 << 24) << " elements (~67 MB)\n";
    std::cout << std::string(90, '-') << "\n";

    for (int bs : {32, 64, 128, 256, 512, 1024}) {
        auto res = benchmarkVector(bs, iterations, 1 << 24);
        printResult(bs, res, iterations);
    }

    printHeader("MATRIX MULTIPLICATION (Naive)");
    std::cout << "Size: 1024×1024 × 1024×1024\n";
    std::cout << std::string(90, '-') << "\n";

    for (int bs : {8, 16}) {
        auto res = benchmarkMatrix(bs, 5, 1024, 1024, 1024, false);
        printResult(bs, res, 5);
    }

    printHeader("MATRIX MULTIPLICATION (Optimized - Tiled)");
    std::cout << "Size: 1024×1024 × 1024×1024\n";
    std::cout << std::string(90, '-') << "\n";

    for (int bs : {32}) {  // 32 works best for TILE_SIZE=32
        auto res = benchmarkMatrix(bs, 5, 1024, 1024, 1024, true);
        printResult(bs, res, 5);
    }

    std::cout << "\n" << std::string(90, '=') << "\n";

    return 0;
}
