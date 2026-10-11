// matmul_all.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define BLOCKSIZE 32
// 一维 Thread Tile：每个线程在 N 方向负责的输出元素个数
#define TM_1D 4
// 二维 Thread Tile：每个线程在 M / N 方向负责的输出元素个数
#define TM_2D 4
#define TN_2D 4

// ---------------------------------------------------------------------------
// 统一的 CUDA 错误检查：出错打印并退出，不在函数里 return
// ---------------------------------------------------------------------------
inline void check_cuda(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

// ===========================================================================
// 1. 朴素版本
// ===========================================================================
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

// ===========================================================================
// 2. 共享内存分块版本
// ===========================================================================
__global__ void matmul_shared_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    const int gRow = cRow * BLOCKSIZE + threadRow;
    const int gCol = cCol * BLOCKSIZE + threadCol;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        As[threadRow * BLOCKSIZE + threadCol] =
            (gRow < M && bkIdx + threadCol < K)
                ? A[threadRow * K + threadCol]
                : 0.0f;

        Bs[threadRow * BLOCKSIZE + threadCol] =
            (bkIdx + threadRow < K && gCol < N)
                ? B[threadRow * N + threadCol]
                : 0.0f;

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    if (gRow < M && gCol < N)
    {
        C[threadRow * N + threadCol] = tmp;
    }
}

// ===========================================================================
// 3. 一维 Thread Tile 版本
// ===========================================================================
__global__ void matmul_shared_1d_tile_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    const int gRow = cRow * BLOCKSIZE + threadRow;
    const int gCol = cCol * BLOCKSIZE * TM_1D + threadCol;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * (BLOCKSIZE * TM_1D)];

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE * TM_1D;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE * TM_1D;

    float tmp[TM_1D] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        As[threadRow * BLOCKSIZE + threadCol] =
            (gRow < M && bkIdx + threadCol < K)
                ? A[threadRow * K + threadCol]
                : 0.0f;

        for (int i = 0; i < TM_1D; ++i)
        {
            const int bCol = threadCol + i * BLOCKSIZE;

            Bs[threadRow * (BLOCKSIZE * TM_1D) + bCol] =
                (bkIdx + threadRow < K && gCol + i * BLOCKSIZE < N)
                    ? B[threadRow * N + bCol]
                    : 0.0f;
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            const float aVal = As[threadRow * BLOCKSIZE + dotIdx];

            for (int i = 0; i < TM_1D; ++i)
            {
                tmp[i] += aVal *
                          Bs[dotIdx * (BLOCKSIZE * TM_1D) + threadCol + i * BLOCKSIZE];
            }
        }

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    for (int i = 0; i < TM_1D; ++i)
    {
        if (gRow < M && gCol + i * BLOCKSIZE < N)
        {
            C[threadRow * N + threadCol + i * BLOCKSIZE] = tmp[i];
        }
    }
}

// ===========================================================================
// 4. 二维 Thread Tile 版本
// ===========================================================================
__global__ void matmul_shared_2d_tile_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    const int gRow = cRow * BLOCKSIZE * TM_2D + threadRow * TM_2D;
    const int gCol = cCol * BLOCKSIZE * TN_2D + threadCol * TN_2D;

    __shared__ float As[BLOCKSIZE * TM_2D * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE * TN_2D];

    A += cRow * BLOCKSIZE * TM_2D * K;
    B += cCol * BLOCKSIZE * TN_2D;
    C += cRow * BLOCKSIZE * TM_2D * N + cCol * BLOCKSIZE * TN_2D;

    float tmp[TM_2D][TN_2D];
    for (int i = 0; i < TM_2D; ++i)
    {
        for (int j = 0; j < TN_2D; ++j)
        {
            tmp[i][j] = 0.0f;
        }
    }

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        for (int i = 0; i < TM_2D; ++i)
        {
            const int aRow = threadRow + i * BLOCKSIZE;

            As[aRow * BLOCKSIZE + threadCol] =
                (gRow + i * BLOCKSIZE < M && bkIdx + threadCol < K)
                    ? A[aRow * K + threadCol]
                    : 0.0f;
        }

        for (int j = 0; j < TN_2D; ++j)
        {
            const int bCol = threadCol + j * BLOCKSIZE;

            Bs[threadRow * (BLOCKSIZE * TN_2D) + bCol] =
                (bkIdx + threadRow < K && gCol + j * BLOCKSIZE < N)
                    ? B[threadRow * N + bCol]
                    : 0.0f;
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            float aReg[TM_2D];
            for (int i = 0; i < TM_2D; ++i)
            {
                aReg[i] = As[(threadRow + i * BLOCKSIZE) * BLOCKSIZE + dotIdx];
            }

            float bReg[TN_2D];
            for (int j = 0; j < TN_2D; ++j)
            {
                bReg[j] = Bs[dotIdx * (BLOCKSIZE * TN_2D) + threadCol + j * BLOCKSIZE];
            }

            for (int i = 0; i < TM_2D; ++i)
            {
                for (int j = 0; j < TN_2D; ++j)
                {
                    tmp[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    for (int i = 0; i < TM_2D; ++i)
    {
        for (int j = 0; j < TN_2D; ++j)
        {
            if (gRow + i * BLOCKSIZE < M && gCol + j * BLOCKSIZE < N)
            {
                C[(threadRow + i * BLOCKSIZE) * N + threadCol + j * BLOCKSIZE] =
                    tmp[i][j];
            }
        }
    }
}

// ===========================================================================
// CPU 参考实现
// ===========================================================================
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

// ===========================================================================
// 最大误差
// ===========================================================================
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

// ===========================================================================
// 通用计时辅助：跑 repeat 次 kernel，返回平均毫秒
// 出错时返回 -1.0f
// ===========================================================================
template <typename KernelLauncher>
float benchmark(KernelLauncher launch, int repeat, cudaEvent_t start, cudaEvent_t stop)
{
    // 先跑一次，确保 kernel 已加载、缓存已预热
    launch();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error in launch: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error in sync: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    err = cudaEventRecord(start);
    if (err != cudaSuccess)
    {
        printf("CUDA error in event record: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    for (int i = 0; i < repeat; ++i)
    {
        launch();
    }

    err = cudaEventRecord(stop);
    if (err != cudaSuccess)
    {
        printf("CUDA error in event record: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
    {
        printf("CUDA error in event sync: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    float ms = 0.0f;
    err = cudaEventElapsedTime(&ms, start, stop);
    if (err != cudaSuccess)
    {
        printf("CUDA error in elapsed time: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }

    return ms / repeat;
}

// ===========================================================================
// 对单个矩阵尺寸跑四个版本，输出结果和相对 naive 的加速比
// ===========================================================================
void run_one_size(int M, int N, int K, int repeat)
{
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
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));

    // 5. Host -> Device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // 6. CPU 参考结果
    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double flops = 2.0 * M * N * K;

    printf("==============================================================\n");
    printf("Matrix size: M = %d, N = %d, K = %d, repeat = %d\n", M, N, K, repeat);
    printf("==============================================================\n");
    printf("%-12s %12s %12s %12s %14s\n",
           "version", "time(ms)", "GFLOPS", "speedup", "max error");
    printf("--------------------------------------------------------------\n");

    // 保存 naive 的时间，用于后面算加速比
    float base_ms = 0.0f;

    // -----------------------------------------------------------------------
    // 7. 朴素版本（base）
    // -----------------------------------------------------------------------
    {
        dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
        dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

        auto launch = [&]() {
            matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        };

        base_ms = benchmark(launch, repeat, start, stop);

        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        printf("%-12s %12.3f %12.2f %12s %14e\n",
               "naive",
               base_ms,
               flops / (base_ms * 1e6),
               "1.00x",
               max_error(h_C, h_C_ref, M * N));
    }

    // -----------------------------------------------------------------------
    // 8. 共享内存版本
    // -----------------------------------------------------------------------
    {
        dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
        dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

        auto launch = [&]() {
            matmul_shared_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        };

        float ms = benchmark(launch, repeat, start, stop);

        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        printf("%-12s %12.3f %12.2f %11.2fx %14e\n",
               "shared",
               ms,
               flops / (ms * 1e6),
               base_ms / ms,
               max_error(h_C, h_C_ref, M * N));
    }

    // -----------------------------------------------------------------------
    // 9. 一维 Thread Tile 版本
    // -----------------------------------------------------------------------
    {
        dim3 gridDim(CEIL_DIV(N, BLOCKSIZE * TM_1D), CEIL_DIV(M, BLOCKSIZE));
        dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

        auto launch = [&]() {
            matmul_shared_1d_tile_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        };

        float ms = benchmark(launch, repeat, start, stop);

        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        printf("%-12s %12.3f %12.2f %11.2fx %14e\n",
               "1d tile",
               ms,
               flops / (ms * 1e6),
               base_ms / ms,
               max_error(h_C, h_C_ref, M * N));
    }

    // -----------------------------------------------------------------------
    // 10. 二维 Thread Tile 版本
    // -----------------------------------------------------------------------
    {
        dim3 gridDim(CEIL_DIV(N, BLOCKSIZE * TN_2D), CEIL_DIV(M, BLOCKSIZE * TM_2D));
        dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

        auto launch = [&]() {
            matmul_shared_2d_tile_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        };

        float ms = benchmark(launch, repeat, start, stop);

        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        printf("%-12s %12.3f %12.2f %11.2fx %14e\n",
               "2d tile",
               ms,
               flops / (ms * 1e6),
               base_ms / ms,
               max_error(h_C, h_C_ref, M * N));
    }

    printf("--------------------------------------------------------------\n\n");

    // 11. 释放资源
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
}

int main()
{
    // 要测试的矩阵尺寸列表（方阵）
    const int sizes[] = {256, 512, 1024, 2048};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    // 每个尺寸重复跑的次数
    const int repeat = 20;

    for (int s = 0; s < numSizes; ++s)
    {
        int size = sizes[s];
        run_one_size(size, size, size, repeat);
    }

    return 0;
}