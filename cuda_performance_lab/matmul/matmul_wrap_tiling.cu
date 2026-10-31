// matmul_shared_2d_tile_warp.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// 配置（自洽校验见文末）
// ---------------------------------------------------------------------------
#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

#define WM 64
#define WN 128
#define WMITER 2          // WSUBM = 32
#define WNITER 2          // WSUBN = 64

#define WSUBM (WM / WMITER)   // 32
#define WSUBN (WN / WNITER)   // 64

#define WARPSIZE 32
#define NUM_WARPS ((BM / WM) * (BN / WN))          // 2
#define NUM_THREADS (NUM_WARPS * WARPSIZE)         // 64

// 编译期自检
static_assert((WSUBM / TM) * (WSUBN / TN) == WARPSIZE,
              "WSUBM/TM * WSUBN/TN must equal WARPSIZE");
static_assert(BM % WM == 0 && BN % WN == 0,
              "BM % WM and BN % WN must be 0");
static_assert(WM % WMITER == 0 && WN % WNITER == 0,
              "WM % WMITER and WN % WNITER must be 0");

// float4
#define VEC 4
#define LDG_A_NUM (BM * BK / NUM_THREADS / VEC)
#define LDG_B_NUM (BK * BN / NUM_THREADS / VEC)
#define STRIDE_A  (BM / LDG_A_NUM)
#define STRIDE_B  (BK / LDG_B_NUM)

static_assert((BM * BK) % (NUM_THREADS * VEC) == 0, "A load not divisible");
static_assert((BK * BN) % (NUM_THREADS * VEC) == 0, "B load not divisible");

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define LDG_FLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])
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
// Warp tiling kernel
// ---------------------------------------------------------------------------
__global__ void matmul_shared_2d_tile_warp_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    // ---- Warp 坐标 ----
    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);

    // ---- Warp 内线程坐标 ----
    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    // ---- 加载坐标（float4 对齐）----
    const uint inner_row_a = threadIdx.x / (BK / VEC);
    const uint inner_col_a = (threadIdx.x % (BK / VEC)) * VEC;

    const uint inner_row_b = threadIdx.x / (BN / VEC);
    const uint inner_col_b = (threadIdx.x % (BN / VEC)) * VEC;

    __shared__ float smem_a[BK * BM];   // 转置 [BK][BM]
    __shared__ float smem_b[BK * BN];   //       [BK][BN]

    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    float thread_results[WMITER * TM * WNITER * TN];
    #pragma unroll
    for (int i = 0; i < WMITER * TM * WNITER * TN; ++i)
        thread_results[i] = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // ---- 加载 A（转置写 smem_a）----
        #pragma unroll
        for (int i = 0; i < BM; i += STRIDE_A)
        {
            float ldg_reg_a[VEC];
            STG_FLOAT4(ldg_reg_a[0]) =
                LDG_FLOAT4(A[OFFSET(i + inner_row_a, inner_col_a, K)]);

            smem_a[OFFSET(inner_col_a + 0, i + inner_row_a, BM)] = ldg_reg_a[0];
            smem_a[OFFSET(inner_col_a + 1, i + inner_row_a, BM)] = ldg_reg_a[1];
            smem_a[OFFSET(inner_col_a + 2, i + inner_row_a, BM)] = ldg_reg_a[2];
            smem_a[OFFSET(inner_col_a + 3, i + inner_row_a, BM)] = ldg_reg_a[3];
        }

        // ---- 加载 B（float4 直通）----
        #pragma unroll
        for (int i = 0; i < BK; i += STRIDE_B)
        {
            STG_FLOAT4(smem_b[OFFSET(inner_row_b + i, inner_col_b, BN)]) =
                LDG_FLOAT4(B[OFFSET(inner_row_b + i, inner_col_b, N)]);
        }

        __syncthreads();

        // ---- 外积：warp tiling ----
        #pragma unroll
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            // 本线程需要的 A 值：WMITER 组，每组 TM 个
            // 行号 = warp_row*WM + ws*WSUBM + thread_row_in_warp*TM + m
            float reg_a[WMITER * TM];
            #pragma unroll
            for (uint ws = 0; ws < WMITER; ++ws)
            {
                const int row_base =
                    warp_row * WM + ws * WSUBM + thread_row_in_warp * TM;
                #pragma unroll
                for (int m = 0; m < TM; m += VEC)
                {
                    STG_FLOAT4(reg_a[ws * TM + m]) =
                        LDG_FLOAT4(smem_a[OFFSET(dot_idx, row_base + m, BM)]);
                }
            }

            // 本线程需要的 B 值：WNITER 组，每组 TN 个
            float reg_b[WNITER * TN];
            #pragma unroll
            for (uint ws = 0; ws < WNITER; ++ws)
            {
                const int col_base =
                    warp_col * WN + ws * WSUBN + thread_col_in_warp * TN;
                #pragma unroll
                for (int n = 0; n < TN; n += VEC)
                {
                    STG_FLOAT4(reg_b[ws * TN + n]) =
                        LDG_FLOAT4(smem_b[OFFSET(dot_idx, col_base + n, BN)]);
                }
            }

            // 外积累加
            #pragma unroll
            for (uint wr = 0; wr < WMITER; ++wr)
            {
                #pragma unroll
                for (uint wc = 0; wc < WNITER; ++wc)
                {
                    #pragma unroll
                    for (int m = 0; m < TM; ++m)
                    {
                        #pragma unroll
                        for (int n = 0; n < TN; ++n)
                        {
                            thread_results[(wr * TM + m) * (WNITER * TN) +
                                           (wc * TN + n)] +=
                                reg_a[wr * TM + m] * reg_b[wc * TN + n];
                        }
                    }
                }
            }
        }

        __syncthreads();

        A += BK;
        B += BK * N;
    }

    // ---- 写回 C（warp tiling）----
    #pragma unroll
    for (uint wr = 0; wr < WMITER; ++wr)
    {
        #pragma unroll
        for (uint wc = 0; wc < WNITER; ++wc)
        {
            float* C_interim = C +
                (warp_row * WM + wr * WSUBM) * N +
                (warp_col * WN + wc * WSUBN);

            #pragma unroll
            for (int m = 0; m < TM; ++m)
            {
                #pragma unroll
                for (int n = 0; n < TN; n += VEC)
                {
                    STG_FLOAT4(C_interim[OFFSET(thread_row_in_warp * TM + m,
                                                thread_col_in_warp * TN + n,
                                                N)]) =
                        LDG_FLOAT4(thread_results[(wr * TM + m) * (WNITER * TN) +
                                                  (wc * TN + n)]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU 参考
// ---------------------------------------------------------------------------
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
}

float max_error(const float* C, const float* C_ref, int size)
{
    float maxError = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        float e = fabsf(C[i] - C_ref[i]);
        if (e > maxError) maxError = e;
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

    matmul_shared_2d_tile_warp_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i)
        matmul_shared_2d_tile_warp_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeat;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    printf("warp tile (BM=%d BN=%d BK=%d TM=%d TN=%d WM=%d WN=%d WMITER=%d WNITER=%d): "
           "%8.3f ms, %7.2f GFLOPS, max error = %e\n",
           BM, BN, BK, TM, TN, WM, WN, WMITER, WNITER,
           ms, flops / (ms * 1e6), max_error(h_C, h_C_ref, M * N));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    return 0;
}