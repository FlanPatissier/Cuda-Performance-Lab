// reduce_bench.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// block 大小（模板参数，必须是编译期常量）
#define BLOCKSIZE 256

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
// GPU 版本 1：naive 版本
//   每个 block 处理 BLOCKSIZE 个元素，共享内存树形归约
//   越界元素填 0，保证共享内存无脏数据
// ===========================================================================
template <int BS>
__global__ void reduce_naive_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];

    const int tid = threadIdx.x;
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 搬入共享内存，越界补 0
    sdata[tid] = (i < len) ? arr[i] : 0;
    __syncthreads();

    // 2. 树形归约，每轮活跃线程减半

    for (int s = 1; s < BS; s <<= 1)
    {
        if ((tid & (2 * s - 1)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }


    // 3. block 内 0 号线程写出部分和
    if (tid == 0)
    {
        out[blockIdx.x] = sdata[0];
    }
}

// ===========================================================================
// GPU 版本 2：grid-stride 版本（推荐）
//   每个线程按网格步长处理多个元素，先做局部累加，再做块内归约
//   优点：固定网格大小即可覆盖任意长度数据，减少 block 数量
// ===========================================================================
template <int BS>
__global__ void reduce_grid_stride_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];

    const int tid    = threadIdx.x;
    const int i      = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // 1. 每个线程先把自己负责的若干元素累加成一个局部和
    int local_sum = 0;

    for (int k = i; k < len; k += stride)
    {
        local_sum += arr[k];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // 2. 块内树形归约（从最大步长开始，活跃线程逐轮减半）
    for (int s = BS / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 写出部分和
    if (tid == 0)
    {
        out[blockIdx.x] = sdata[0];
    }
}

// ===========================================================================
// 封装：多级归约直到只剩 1 个结果，全部在 GPU 上完成
// 返回最终的 sum
// ===========================================================================
int reduce_gpu_multilevel(const int* d_arr, int len, bool use_grid_stride)
{
    int gridsize = CEIL_DIV(len, BLOCKSIZE);

    int* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, sizeof(int) * gridsize));

    // ---- 第一级 ----
    if (use_grid_stride)
    {
        // 固定网格大小，用 grid-stride 覆盖全部数据
        int fixed_grid = 128;
        reduce_grid_stride_kernel<BLOCKSIZE><<<fixed_grid, BLOCKSIZE>>>(d_arr, d_buf, len);
        gridsize = fixed_grid;
    }
    else
    {
        reduce_naive_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_buf, len);
    }

    // ---- 后续各级：迭代归约直到只剩 1 个 ----
    int n = gridsize;
    int* d_in = d_buf;

    while (n > 1)
    {
        int next_grid = CEIL_DIV(n, BLOCKSIZE);

        int* d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(int) * next_grid));

        reduce_naive_kernel<BLOCKSIZE><<<next_grid, BLOCKSIZE>>>(d_in, d_tmp, n);

        if (d_in != d_buf)
        {
            cudaFree(d_in);
        }

        d_in = d_tmp;
        n    = next_grid;
    }

    int result = 0;
    CUDA_CHECK(cudaMemcpy(&result, d_in, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_in);

    if (d_in != d_buf)
    {
        cudaFree(d_buf);
    }

    return result;
}

// ===========================================================================
// 主函数
// ===========================================================================
int main()
{
    // 数据规模（可改成其他值，比如 1e7、1e8）
    int len = 100000000;

    size_t bytes = sizeof(int) * len;

    // 1. 申请主机内存并初始化
    int* h_arr = new int[len];

    for (int i = 0; i < len; ++i)
    {
        h_arr[i] = 1;   // 全部为 1，方便校验：sum 应等于 len
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

    // CPU 用标准库高精度计时
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    int cpu_sum = reduce_cpu(h_arr, len);

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double cpu_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 +
                    (ts1.tv_nsec - ts0.tv_nsec) / 1e6;

    printf("CPU sum = %d, time = %.3f ms\n", cpu_sum, cpu_ms);

    const int repeat = 20;

    // ------------------------------------------------------------------
    // 5. naive 版本
    // ------------------------------------------------------------------
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

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;

        // 把部分和拷回，CPU 做最后归约
        int* h_out = new int[gridsize];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * gridsize,
                              cudaMemcpyDeviceToHost));

        int gpu_sum = 0;
        for (int i = 0; i < gridsize; ++i)
        {
            gpu_sum += h_out[i];
        }

        printf("naive  : %8.3f ms, sum = %d, %s\n",
               ms, gpu_sum, (gpu_sum == cpu_sum) ? "success" : "failed");

        delete[] h_out;
        cudaFree(d_out);
    }

    // ------------------------------------------------------------------
    // 6. grid-stride 版本
    // ------------------------------------------------------------------
    {
        int gridsize = 128;   // 固定网格大小
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * gridsize));

        reduce_grid_stride_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
        {
            reduce_grid_stride_kernel<BLOCKSIZE><<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;

        int* h_out = new int[gridsize];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * gridsize,
                              cudaMemcpyDeviceToHost));

        int gpu_sum = 0;
        for (int i = 0; i < gridsize; ++i)
        {
            gpu_sum += h_out[i];
        }

        printf("gridstr: %8.3f ms, sum = %d, %s\n",
               ms, gpu_sum, (gpu_sum == cpu_sum) ? "success" : "failed");

        delete[] h_out;
        cudaFree(d_out);
    }

    // ------------------------------------------------------------------
    // 7. 多级归约版本（含 H2D 之外的完整 GPU 归约）
    // ------------------------------------------------------------------
    {
        // warmup
        reduce_gpu_multilevel(d_arr, len, false);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
        {
            reduce_gpu_multilevel(d_arr, len, false);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;

        int gpu_sum = reduce_gpu_multilevel(d_arr, len, false);

        printf("multilv: %8.3f ms, sum = %d, %s\n",
               ms, gpu_sum, (gpu_sum == cpu_sum) ? "success" : "failed");
    }

    // ------------------------------------------------------------------
    // 8. 打印对比
    // ------------------------------------------------------------------
    printf("\n===== Summary =====\n");
    printf("Data size : %d ints (%.2f MB)\n", len, bytes / (1024.0 * 1024.0));
    printf("CPU time  : %.3f ms\n", cpu_ms);

    // 9. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_arr));

    delete[] h_arr;

    return 0;
}