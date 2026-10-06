// matmul_shared.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define BLOCKSIZE 32

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t error = call;                                  \
        if (error != cudaSuccess)                                  \
        {                                                          \
            printf("CUDA error: %s\n", cudaGetErrorString(error)); \
            return 1;                                              \
        }                                                          \
    } while (0)

// 朴素版本，作为性能对照
// A: M x K
// B: K x N
// C: M x N
__global__ void matmul_naive_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// 共享内存分块版本
// 一个 block 负责 C 中一块 BLOCKSIZE x BLOCKSIZE 的子矩阵
// block 内用一维线程组织，共 BLOCKSIZE * BLOCKSIZE 个线程
__global__ void matmul_shared_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // 当前 block 负责的输出块坐标
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // threadCol 取 threadIdx.x 的低位，保证同一个 warp 内 threadCol 连续
    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    // 该线程最终负责的 C 元素的全局坐标
    const int gRow = cRow * BLOCKSIZE + threadRow;
    const int gCol = cCol * BLOCKSIZE + threadCol;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // 推进指针到起始位置
    A += cRow * BLOCKSIZE * K;                    // 行 = cRow 块，列 = 0
    B += cCol * BLOCKSIZE;                        // 行 = 0，列 = cCol 块
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // 行 = cRow 块，列 = cCol 块

    float tmp = 0.0f;

    // 外层循环沿 K 方向推进：A 向右滑动，B 向下滑动
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        // 每个线程从全局内存各搬运 A、B 的一个元素到共享内存
        // 越界的位置补 0，这样 M/N/K 不是 BLOCKSIZE 的整数倍时结果依然正确
        As[threadRow * BLOCKSIZE + threadCol] =
            (gRow < M && bkIdx + threadCol < K)
                ? A[threadRow * K + threadCol]
                : 0.0f;

        Bs[threadRow * BLOCKSIZE + threadCol] =
            (bkIdx + threadRow < K && gCol < N)
                ? B[threadRow * N + threadCol]
                : 0.0f;

        // 等所有线程都把自己那份数据写进共享内存
        __syncthreads();

        // 在当前缓存块上做部分点积
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // 再同步一次，防止跑得快的线程提前覆盖还在被读的共享内存
        __syncthreads();

        // 推进指针到下一个块
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    if (gRow < M && gCol < N)
    {
        C[threadRow * N + threadCol] = tmp;
    }
}

void cpu_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

float max_error(const float* C, const float* C_ref, int size)
{
    float maxError = 0.0f;

    for (int i = 0; i < size; ++i)
    {
        float error = fabsf(C[i] - C_ref[i]);

        if (error > maxError)
        {
            maxError = error;
        }
    }

    return maxError;
}

int main()
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    // 1. 申请主机内存
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_C_ref = new float[M * N];

    // 2. 初始化矩阵
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = ((i % 7) - 3) / 7.0f;
    }

    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = ((i % 5) - 2) / 5.0f;
    }

    // 3. 定义设备指针
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    // 4. 申请 GPU 内存
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // 5. Host -> Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // 6. CPU 参考结果
    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int repeat = 20;
    double flops = 2.0 * M * N * K;

    dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));

    // 7. 朴素版本
    {
        dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

        matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
        {
            matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;

        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        printf("naive  : %8.3f ms, %7.2f GFLOPS, max error = %e\n",
               ms,
               flops / (ms * 1e6),
               max_error(h_C, h_C_ref, M * N));
    }

    // 8. 共享内存版本
    {
        dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

        matmul_shared_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
        {
            matmul_shared_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;

        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        printf("shared : %8.3f ms, %7.2f GFLOPS, max error = %e\n",
               ms,
               flops / (ms * 1e6),
               max_error(h_C, h_C_ref, M * N));
    }

    // 9. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return 0;
}
