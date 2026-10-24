// matmul_shared_2d_tile_vec.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// 配置（要求 BM/BN/BK/TM/TN 均为 4 的倍数，以支持 float4）
// ---------------------------------------------------------------------------
#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

#define NUM_THREADS ((BM * BN) / (TM * TN))   // 256
#define THREAD_COLS (BN / TN)                 // 16

// float4 相关
#define VEC 4
#define LDG_A_NUM (BM * BK / NUM_THREADS / VEC)
#define LDG_B_NUM (BK * BN / NUM_THREADS / VEC)
#define STRIDE_A  (BM / LDG_A_NUM)
#define STRIDE_B  (BK / LDG_B_NUM)

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 读：const 安全，返回 float4 值
#define LDG_FLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])
// 写：返回 float4 左值引用
#define STG_FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

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
// 向量化访存 + 二维 Thread Tile
//   smem_a: [BK][BM]  —— 转置存放，行宽 BM
//   smem_b: [BK][BN]  —— 不转置，行宽 BN
// ---------------------------------------------------------------------------
__global__ void matmul_shared_2d_tile_vec_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    // Block / Thread 坐标
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    const int block_row_thread = BN / TN;    // 线程列数
    const int block_col_thread = BM / TM;    // 线程行数
    (void)block_col_thread;                  // 避免 unused 警告

    // 线程在 block 内的 thread tile 左上角（已乘 TM/TN）
    const int thread_col = (threadIdx.x % block_row_thread) * TN;
    const int thread_row = (threadIdx.x / block_row_thread) * TM;

    // 加载坐标（按 float4 对齐）
    const uint inner_row_a = threadIdx.x / (BK / VEC);
    const uint inner_col_a = (threadIdx.x % (BK / VEC)) * VEC;   // 0,4,8,12

    const uint inner_row_b = threadIdx.x / (BN / VEC);
    const uint inner_col_b = (threadIdx.x % (BN / VEC)) * VEC;

    // 共享内存
    __shared__ float smem_a[BK * BM];   // 转置后布局：[BK][BM]
    __shared__ float smem_b[BK * BN];   // [BK][BN]

    // 推进到当前 block 的起点
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    // 每线程 TM x TN 累加器
    float thread_results[TM * TN];
    #pragma unroll
    for (int i = 0; i < TM * TN; ++i)
        thread_results[i] = 0.0f;

    // 沿 K 方向迭代
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // -------------------------------------------------------------------
        // 1) 加载 A 的 BM x BK tile 到 smem_a（转置存放）
        // -------------------------------------------------------------------
        {
            float ldg_reg_a[LDG_A_NUM * VEC];

            #pragma unroll
            for (int i = 0; i < BM; i += STRIDE_A)
            {
                const int ldg_index = i / STRIDE_A * VEC;

                // 读 global（const 安全）
                STG_FLOAT4(ldg_reg_a[ldg_index]) =
                    LDG_FLOAT4(A[OFFSET(i + inner_row_a, inner_col_a, K)]);

                // 转置写 shared（标量，列方向跨行，无法 float4 写）
                smem_a[OFFSET(inner_col_a + 0, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 0];
                smem_a[OFFSET(inner_col_a + 1, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 1];
                smem_a[OFFSET(inner_col_a + 2, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 2];
                smem_a[OFFSET(inner_col_a + 3, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 3];
            }
        }

        // -------------------------------------------------------------------
        // 2) 加载 B 的 BK x BN tile 到 smem_b（不转置，向量写）
        // -------------------------------------------------------------------
        {
            #pragma unroll
            for (int i = 0; i < BK; i += STRIDE_B)
            {
                STG_FLOAT4(smem_b[OFFSET(inner_row_b + i, inner_col_b, BN)]) =
                    LDG_FLOAT4(B[OFFSET(inner_row_b + i, inner_col_b, N)]);
            }
        }

        __syncthreads();

        // -------------------------------------------------------------------
        // 3) 外积累加（从 smem 用 float4 读 reg_a/reg_b）
        // -------------------------------------------------------------------
        #pragma unroll
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            float reg_a[TM];
            #pragma unroll
            for (int m = 0; m < TM; m += VEC)
            {
                STG_FLOAT4(reg_a[m]) =
                    LDG_FLOAT4(smem_a[OFFSET(dot_idx, thread_row + m, BM)]);
            }

            float reg_b[TN];
            #pragma unroll
            for (int n = 0; n < TN; n += VEC)
            {
                STG_FLOAT4(reg_b[n]) =
                    LDG_FLOAT4(smem_b[OFFSET(dot_idx, thread_col + n, BN)]);
            }

            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    thread_results[m * TN + n] += reg_a[m] * reg_b[n];
        }

        __syncthreads();

        A += BK;
        B += BK * N;
    }

    // -----------------------------------------------------------------------
    // 4) 向量化写回 C
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int m = 0; m < TM; ++m)
    {
        #pragma unroll
        for (int n = 0; n < TN; n += VEC)
        {
            STG_FLOAT4(C[OFFSET(thread_row + m, thread_col + n, N)]) =
                LDG_FLOAT4(thread_results[m * TN + n]);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU 参考
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
                sum += A[row * K + k] * B[k * N + col];
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
        if (error > maxError) maxError = error;
    }
    return maxError;
}

int main()
{
    int M = 1024, N = 1024, K = 1024;

    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_C_ref = new float[M * N];

    for (int i = 0; i < M * K; ++i) h_A[i] = ((i % 7) - 3) / 7.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = ((i % 5) - 2) / 5.0f;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int repeat = 20;
    double flops = 2.0 * M * N * K;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(NUM_THREADS);

    matmul_shared_2d_tile_vec_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i)
        matmul_shared_2d_tile_vec_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeat;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    printf("vec 2d tile (BM=%d BN=%d BK=%d TM=%d TN=%d): %8.3f ms, %7.2f GFLOPS, max error = %e\n",
           BM, BN, BK, TM, TN,
           ms,
           flops / (ms * 1e6),
           max_error(h_C, h_C_ref, M * N));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    return 0;
}