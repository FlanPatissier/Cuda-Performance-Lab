// matmul_shared_2d_tile.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
//   BM : block tile 在 M 方向输出行数
//   BN : block tile 在 N 方向输出列数
//   BK : K 方向每次 shared-memory 迭代深度
//   TM : 每线程 M 方向输出行数
//   TN : 每线程 N 方向输出列数
//
// 约束：
//   (BM*BN) % (TM*TN) == 0     -> number_threads
//   BM % TM == 0, BN % TN == 0
//   number_threads % BK == 0   -> stride_A 整除
//   number_threads % BN == 0   -> stride_B 整除
// ---------------------------------------------------------------------------
#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

#define NUM_THREADS ((BM * BN) / (TM * TN))   // 256
#define THREAD_COLS (BN / TN)                 // 16
#define STRIDE_A    (NUM_THREADS / BK)
#define STRIDE_B    (NUM_THREADS / BN)

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
// 二维 Thread Tile + 五参数解耦版本
//   - 加载索引（inner_row_*/inner_col_*/stride_*）与计算坐标解耦
//   - BM/BN/BK/TM/TN 独立可调，M/N 方向 tile 可不对称
// ---------------------------------------------------------------------------
__global__ void matmul_shared_2d_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    // -----------------------------------------------------------------------
    // Block 索引
    // -----------------------------------------------------------------------
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    // -----------------------------------------------------------------------
    // Thread 索引（用于计算）：把 block tile 按 (BN/TN) 列切分
    // -----------------------------------------------------------------------
    const uint thread_col = threadIdx.x % THREAD_COLS;
    const uint thread_row = threadIdx.x / THREAD_COLS;

    // -----------------------------------------------------------------------
    // 加载索引（与计算坐标解耦，扁平展开）
    //   A tile: BM x BK , 行宽 BK
    //   B tile: BK x BN , 行宽 BN
    // -----------------------------------------------------------------------
    const uint inner_row_A = threadIdx.x / BK;   // [0, BM)
    const uint inner_col_A = threadIdx.x % BK;   // [0, BK)

    const uint inner_row_B = threadIdx.x / BN;   // [0, BK)
    const uint inner_col_B = threadIdx.x % BN;   // [0, BN)

    // -----------------------------------------------------------------------
    // 该线程负责的 TM x TN 输出小块左上角全局坐标
    // -----------------------------------------------------------------------
    const uint gRow = c_row * BM + thread_row * TM;
    const uint gCol = c_col * BN + thread_col * TN;

    // -----------------------------------------------------------------------
    // 共享内存
    // -----------------------------------------------------------------------
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 把 A/B/C 指针推进到当前 block 的起始位置
    A += c_row * BM * K;   // 行偏移，列从 0 起（每次迭代再加 bkIdx）
    B += c_col * BN;       // 列偏移，行从 0 起（每次迭代再加 bkIdx*N）
    C += c_row * BM * N + c_col * BN;

    // -----------------------------------------------------------------------
    // 每线程 TM x TN 累加器
    // -----------------------------------------------------------------------
    float tmp[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            tmp[i][j] = 0.0f;

    // -----------------------------------------------------------------------
    // 沿 K 方向迭代
    // -----------------------------------------------------------------------
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // ---- 1) 加载 A 的 BM x BK tile ----
        //     每个线程搬 (BM / STRIDE_A) 个元素（沿 M 方向跨行）
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += STRIDE_A)
        {
            const uint aRow = inner_row_A + offset;
            const uint aCol = inner_col_A;

            const bool valid =
                (c_row * BM + aRow < (uint)M) &&
                (bkIdx + (int)aCol < K);

            As[aRow * BK + aCol] =
                valid ? A[aRow * K + aCol] : 0.0f;
        }

        // ---- 2) 加载 B 的 BK x BN tile ----
        //     每个线程搬 (BK / STRIDE_B) 个元素（沿 K 方向跨行）
        #pragma unroll
        for (uint offset = 0; offset < BK; offset += STRIDE_B)
        {
            const uint bRow = inner_row_B + offset;
            const uint bCol = inner_col_B;

            const bool valid =
                (bkIdx + (int)bRow < K) &&
                (c_col * BN + bCol < (uint)N);

            Bs[bRow * BN + bCol] =
                valid ? B[bRow * N + bCol] : 0.0f;
        }

        __syncthreads();

        // ---- 3) 在当前 tile 上做外积累加 ----
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // 取 TM 个 A 元素
            float aReg[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                aReg[i] = As[(thread_row * TM + i) * BK + dotIdx];

            // 取 TN 个 B 元素
            float bReg[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bReg[j] = Bs[dotIdx * BN + thread_col * TN + j];

            // TM x TN 外积
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    tmp[i][j] += aReg[i] * bReg[j];
        }

        __syncthreads();

        // 推进到下一个 K tile
        A += BK;
        B += BK * N;
    }

    // -----------------------------------------------------------------------
    // 写回 C
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        #pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            const uint row = gRow + i;
            const uint col = gCol + j;
            if (row < (uint)M && col < (uint)N)
                C[(thread_row * TM + i) * N + (thread_col * TN + j)] = tmp[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// CPU 参考实现（不变）
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
// 最大误差（不变）
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

    // grid 按 BM/BN 划分
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    // 7. 二维 Thread Tile 版本
    {
        // block 内线程数 = BM*BN/(TM*TN)
        dim3 blockDim(NUM_THREADS);

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

        printf("2d tile (BM=%d BN=%d BK=%d TM=%d TN=%d): %8.3f ms, %7.2f GFLOPS, max error = %e\n",
               BM, BN, BK, TM, TN,
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