// reduce_bench.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <ctime>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// block 大小（模板参数，必须是编译期常量）
#define BLOCKSIZE 512

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

// ===========================================================================
// CPU 参考实现：串行累加
// ===========================================================================
int reduce_cpu(const int* arr, int len)
{
    int sum = 0;

    for (int i = 0; i < len; ++i)
    {
        sum += arr[i];
    }

    return sum;
}

// ===========================================================================
// GPU 版本 1：基础版（不使用交错寻址）
//   每个 block 处理 BLOCKSIZE 个元素
//   树形归约：活跃线程通过 tid % (2*s) == 0 筛选
// ===========================================================================
template <int BS>
__global__ void reduce_naive_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int bdim = blockDim.x;
    const int i    = bid * bdim + tid;

    // 搬入共享内存（越界补 0，避免读到垃圾值）
    sdata[tid] = (i < len) ? arr[i] : 0;
    __syncthreads();

    // 不使用交错寻址
    for (int s = 1; s < bdim; s <<= 1)
    {
        if ((tid & (2 * s - 1)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[bid] = sdata[0];
    }
}

// ===========================================================================
// GPU 版本 2：交错寻址版（interleaved addressing）
//   index = 2 * s * tid，warp 内地址步长随 s 增大 → 有 bank conflict
// ===========================================================================
template <int BS>
__global__ void reduce_interleaved_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int bdim = blockDim.x;
    const int i    = bid * bdim + tid;

    // 越界补 0
    sdata[tid] = (i < len) ? arr[i] : 0;
    __syncthreads();

    // 交错寻址
    for (int s = 1; s < bdim; s <<= 1)
    {
        int index = 2 * s * tid;

        if (index + s < bdim)
        {
            sdata[index] += sdata[index + s];
        } 
        __syncthreads();
    }

    if (tid == 0)
    {
        out[bid] = sdata[0];
    }
}

// ===========================================================================
// GPU 版本 3：顺序寻址版（sequential addressing）—— 解决 bank conflict
//   s 从 bdim/2 递减到 1，活跃线程 tid < s，
//   访问 sdata[tid] 与 sdata[tid+s]，均为连续地址段 → 无 bank conflict
// ===========================================================================
template <int BS>
__global__ void reduce_sequential_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];

    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int bdim = blockDim.x;
    const int i    = bid * bdim + tid;

    // 越界补 0
    sdata[tid] = (i < len) ? arr[i] : 0;
    __syncthreads();

    // 顺序归约
    for (int s = bdim / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[bid] = sdata[0];
    }
}

// ===========================================================================
// 主函数
// ===========================================================================
int main()
{
    // 数据规模（可改成 1e6、1e7、1e8 观察趋势）
    int len = 100000000;

    size_t bytes = sizeof(int) * len;

    // 1. 申请主机内存并初始化
    int* h_arr = new int[len];

    for (int i = 0; i < len; ++i)
    {
        h_arr[i] = 1;   // 全部为 1，sum 应等于 len
    }

    // 2. 设备指针
    int* d_arr = nullptr;

    CUDA_CHECK(cudaMalloc(&d_arr, bytes));

    // 3. Host -> Device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // 4. CPU 参考结果 + 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    int cpu_sum = reduce_cpu(h_arr, len);

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double cpu_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 +
                    (ts1.tv_nsec - ts0.tv_nsec) / 1e6;

    printf("CPU sum = %d, time = %.3f ms\n", cpu_sum, cpu_ms);

    const int repeat = 20;

    // ------------------------------------------------------------------
    // 5. 基础版（不使用交错寻址）
    // ------------------------------------------------------------------
    float naive_ms = 0.0f;
    {
        int gridsize = CEIL_DIV(len, BLOCKSIZE);
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * gridsize));

        // warmup
        reduce_naive_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
        {
            reduce_naive_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
        naive_ms /= repeat;

        int* h_out = new int[gridsize];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * gridsize,
                              cudaMemcpyDeviceToHost));

        int gpu_sum = 0;
        for (int i = 0; i < gridsize; ++i)
        {
            gpu_sum += h_out[i];
        }

        printf("naive (no interleave): %8.3f ms, sum = %d, %s\n",
               naive_ms, gpu_sum, (gpu_sum == cpu_sum) ? "success" : "failed");

        delete[] h_out;
        cudaFree(d_out);
    }

    // ------------------------------------------------------------------
    // 6. 交错寻址版
    // ------------------------------------------------------------------
    float inter_ms = 0.0f;
    {
        int gridsize = CEIL_DIV(len, BLOCKSIZE);
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * gridsize));

        // warmup
        reduce_interleaved_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
        {
            reduce_interleaved_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&inter_ms, start, stop));
        inter_ms /= repeat;

        int* h_out = new int[gridsize];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * gridsize,
                              cudaMemcpyDeviceToHost));

        int gpu_sum = 0;
        for (int i = 0; i < gridsize; ++i)
        {
            gpu_sum += h_out[i];
        }

        printf("interleaved          : %8.3f ms, sum = %d, %s\n",
               inter_ms, gpu_sum, (gpu_sum == cpu_sum) ? "success" : "failed");

        delete[] h_out;
        cudaFree(d_out);
    }

    // ------------------------------------------------------------------
    // 7. 顺序寻址版（解决 bank conflict）
    // ------------------------------------------------------------------
    float seq_ms = 0.0f;
    {
        int gridsize = CEIL_DIV(len, BLOCKSIZE);
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * gridsize));

        // warmup
        reduce_sequential_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
        {
            reduce_sequential_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&seq_ms, start, stop));
        seq_ms /= repeat;

        int* h_out = new int[gridsize];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * gridsize,
                              cudaMemcpyDeviceToHost));

        int gpu_sum = 0;
        for (int i = 0; i < gridsize; ++i)
        {
            gpu_sum += h_out[i];
        }

        printf("sequential           : %8.3f ms, sum = %d, %s\n",
               seq_ms, gpu_sum, (gpu_sum == cpu_sum) ? "success" : "failed");

        delete[] h_out;
        cudaFree(d_out);
    }

    // ------------------------------------------------------------------
    // 8. 打印对比
    // ------------------------------------------------------------------
    printf("\n===== Summary =====\n");
    printf("Data size       : %d ints (%.2f MB)\n", len, bytes / (1024.0 * 1024.0));
    printf("CPU time        : %.3f ms\n", cpu_ms);
    printf("naive GPU time  : %.3f ms  (speedup vs CPU: %.2fx)\n",
           naive_ms, cpu_ms / naive_ms);
    printf("interleaved time: %.3f ms  (speedup vs CPU: %.2fx)\n",
           inter_ms, cpu_ms / inter_ms);
    printf("sequential time : %.3f ms  (speedup vs CPU: %.2fx)\n",
           seq_ms, cpu_ms / seq_ms);

    // 9. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_arr));

    delete[] h_arr;

    return 0;
}