// matmul_shared_2d_tile.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define BLOCKSIZE 32
// 每个线程在行方向（M 方向）负责的输出元素个数
#define TM 4
// 每个线程在列方向（N 方向）负责的输出元素个数
#define TN 4

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

// ---------------------------------------------------------------------------
// 二维 Thread Tile 版本
// 一个 block 负责 C 中 (BLOCKSIZE * TM) x (BLOCKSIZE * TN) 的一块子矩阵
// block 内使用 BLOCKSIZE * BLOCKSIZE 个线程（一维组织）
// 每个线程负责一个 TM x TN 的小块
// ---------------------------------------------------------------------------
__global__ void matmul_shared_2d_tile_kernel(
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

    // 将一维 threadIdx.x 拆成 (threadRow, threadCol)
    // threadCol 取低位，保证同一个 warp 内 threadCol 连续
    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    // 该线程负责的 TM x TN 小块中，左上角元素的全局坐标
    const int gRow = cRow * BLOCKSIZE * TM + threadRow * TM;
    const int gCol = cCol * BLOCKSIZE * TN + threadCol * TN;

    // 共享内存缓存块
    // As: (BLOCKSIZE * TM) x BLOCKSIZE
    // Bs: BLOCKSIZE x (BLOCKSIZE * TN)
    __shared__ float As[BLOCKSIZE * TM * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE * TN];

    // 推进 A 指针到当前 block 对应的起始位置
    A += cRow * BLOCKSIZE * TM * K;   // 行 = cRow 块，列 = 0
    // 推进 B 指针到当前 block 对应的起始位置
    B += cCol * BLOCKSIZE * TN;       // 行 = 0，列 = cCol 块
    // 推进 C 指针到当前 block 对应的起始位置
    C += cRow * BLOCKSIZE * TM * N + cCol * BLOCKSIZE * TN;

    // 每个线程维护 TM x TN 个部分和
    float tmp[TM][TN];
    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; ++j)
        {
            tmp[i][j] = 0.0f;
        }
    }

    // 外层循环沿 K 方向推进
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        // -------------------------------------------------------------------
        // 1. 把 A 的一个 (BLOCKSIZE * TM) x BLOCKSIZE 块搬进共享内存
        //    每个线程负责搬运 TM 个元素（沿 M 方向）
        //    As 的布局：行优先，行宽 = BLOCKSIZE
        // -------------------------------------------------------------------
        for (int i = 0; i < TM; ++i)
        {
            const int aRow = threadRow + i * BLOCKSIZE;

            As[aRow * BLOCKSIZE + threadCol] =
                (gRow + i * BLOCKSIZE < M && bkIdx + threadCol < K)
                    ? A[aRow * K + threadCol]
                    : 0.0f;
        }

        // -------------------------------------------------------------------
        // 2. 把 B 的一个 BLOCKSIZE x (BLOCKSIZE * TN) 块搬进共享内存
        //    每个线程负责搬运 TN 个元素（沿 N 方向）
        //    Bs 的布局：行优先，行宽 = BLOCKSIZE * TN
        // -------------------------------------------------------------------
        for (int j = 0; j < TN; ++j)
        {
            const int bCol = threadCol + j * BLOCKSIZE;

            Bs[threadRow * (BLOCKSIZE * TN) + bCol] =
                (bkIdx + threadRow < K && gCol + j * BLOCKSIZE < N)
                    ? B[threadRow * N + bCol]
                    : 0.0f;
        }

        // 等所有线程都完成共享内存写入
        __syncthreads();

        // -------------------------------------------------------------------
        // 3. 在当前缓存块上做部分点积
        //    每个线程从 As 读 TM 个元素、从 Bs 读 TN 个元素，
        //    做 TM x TN 次乘加，A 和 B 的读取都只在寄存器层面复用
        // -------------------------------------------------------------------
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            // 从 As 取出该线程负责的 TM 个 A 元素
            float aReg[TM];
            for (int i = 0; i < TM; ++i)
            {
                aReg[i] = As[(threadRow + i * BLOCKSIZE) * BLOCKSIZE + dotIdx];
            }

            // 从 Bs 取出该线程负责的 TN 个 B 元素
            float bReg[TN];
            for (int j = 0; j < TN; ++j)
            {
                bReg[j] = Bs[dotIdx * (BLOCKSIZE * TN) + threadCol + j * BLOCKSIZE];
            }

            // 做 TM x TN 次外积累加
            for (int i = 0; i < TM; ++i)
            {
                for (int j = 0; j < TN; ++j)
                {
                    tmp[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        // 再同步一次，防止跑得快的线程提前覆盖还在被读的共享内存
        __syncthreads();

        // 推进指针到下一个块
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    // -----------------------------------------------------------------------
    // 4. 写回结果
    //    每个线程写 TM x TN 个元素
    // -----------------------------------------------------------------------
    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; ++j)
        {
            if (gRow + i * BLOCKSIZE < M && gCol + j * BLOCKSIZE < N)
            {
                C[(threadRow + i * BLOCKSIZE) * N + threadCol + j * BLOCKSIZE] =
                    tmp[i][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU 参考实现
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// 最大误差
// ---------------------------------------------------------------------------
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

    // 注意：grid 在 M/N 方向的划分要按 BLOCKSIZE * TM / BLOCKSIZE * TN 来算
    dim3 gridDim(CEIL_DIV(N, BLOCKSIZE * TN), CEIL_DIV(M, BLOCKSIZE * TM));

    // 7. 二维 Thread Tile 版本
    {
        // block 内仍然使用 BLOCKSIZE * BLOCKSIZE 个线程（一维）
        dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

        matmul_shared_2d_tile_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
        {
            matmul_shared_2d_tile_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;

        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        printf("2d tile: %8.3f ms, %7.2f GFLOPS, max error = %e\n",
               ms,
               flops / (ms * 1e6),
               max_error(h_C, h_C_ref, M * N));
    }

    // 8. 释放资源
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
