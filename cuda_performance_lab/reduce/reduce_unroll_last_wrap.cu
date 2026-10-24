// reduce_bench.cu —— 五版归约 runtime 对比
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <ctime>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

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
// CPU 参考
// ===========================================================================
int reduce_cpu(const int* arr, int len)
{
    int sum = 0;
    for (int i = 0; i < len; ++i) sum += arr[i];
    return sum;
}

// ===========================================================================
// 版本 1：naive
// ===========================================================================
template <int BS>
__global__ void reduce_naive_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];
    const int tid = threadIdx.x, bid = blockIdx.x, bdim = blockDim.x;
    const int i = bid * bdim + tid;

    sdata[tid] = (i < len) ? arr[i] : 0;
    __syncthreads();

    for (int s = 1; s < bdim; s <<= 1)
    {
        if ((tid & (2 * s - 1)) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[bid] = sdata[0];
}

// ===========================================================================
// 版本 2：interleaved
// ===========================================================================
template <int BS>
__global__ void reduce_interleaved_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];
    const int tid = threadIdx.x, bid = blockIdx.x, bdim = blockDim.x;
    const int i = bid * bdim + tid;

    sdata[tid] = (i < len) ? arr[i] : 0;
    __syncthreads();

    for (int s = 1; s < bdim; s <<= 1)
    {
        int index = 2 * s * tid;
        if (index + s < bdim) sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0) out[bid] = sdata[0];
}

// ===========================================================================
// 版本 3：sequential
// ===========================================================================
template <int BS>
__global__ void reduce_sequential_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];
    const int tid = threadIdx.x, bid = blockIdx.x, bdim = blockDim.x;
    const int i = bid * bdim + tid;

    sdata[tid] = (i < len) ? arr[i] : 0;
    __syncthreads();

    for (int s = bdim / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[bid] = sdata[0];
}

// ===========================================================================
// 版本 4：two-per-thread
// ===========================================================================
template <int BS>
__global__ void reduce_two_per_thread_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];
    const int tid = threadIdx.x, bid = blockIdx.x, bdim = blockDim.x;
    const int i = bid * bdim * 2 + tid;

    int v = 0;
    if (i < len)        v += arr[i];
    if (i + bdim < len) v += arr[i + bdim];
    sdata[tid] = v;
    __syncthreads();

    for (int s = bdim / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[bid] = sdata[0];
}

// ===========================================================================
// warp 内归约辅助函数
//   要求：tid 0..31 全部活跃，且 sdata[0..63] 已完成前几轮累加
//   volatile 强制每次真正读写共享内存，避免寄存器缓存导致 warp 内不一致
// ===========================================================================
__device__ void warp_reduce(volatile int* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// ===========================================================================
// 版本 5：sequential + warp_reduce
//   前几轮仍用 __syncthreads() 的共享内存归约（s > 32）；
//   当 s <= 32 时，活跃线程都在同一个 warp 内，不需要 __syncthreads()，
//   直接用 warp_reduce 做最后 6 步。
// ===========================================================================
template <int BS>
__global__ void reduce_warp_kernel(const int* arr, int* out, int len)
{
    __shared__ int sdata[BS];
    const int tid = threadIdx.x, bid = blockIdx.x, bdim = blockDim.x;
    const int i = bid * bdim + tid;

    sdata[tid] = (i < len) ? arr[i] : 0;
    __syncthreads();

    // 共享内存阶段：s = bdim/2, bdim/4, ..., 64，即 s > 32
    for (int s = bdim / 2; s > 32; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // warp 阶段：tid 0..31 负责最后 6 步（s=32,16,8,4,2,1）
    // 注意循环结束时 s=32 被跳过，正好由 warp_reduce 第一行 sdata[tid]+=sdata[tid+32] 补上
    if (tid < 32)
    {
        warp_reduce(sdata, tid);
    }

    if (tid == 0) out[bid] = sdata[0];
}

// ===========================================================================
// 计时辅助
// ===========================================================================
template <typename Kernel>
float bench_kernel(Kernel kernel, int gridsize, const int* d_arr, int* d_out,
                   int len, cudaEvent_t start, cudaEvent_t stop, int repeat)
{
    kernel<<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i)
        kernel<<<gridsize, BLOCKSIZE>>>(d_arr, d_out, len);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / repeat;
}

int verify(int* d_out, int gridsize)
{
    int* h_out = new int[gridsize];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * gridsize, cudaMemcpyDeviceToHost));
    int s = 0;
    for (int i = 0; i < gridsize; ++i) s += h_out[i];
    delete[] h_out;
    return s;
}

// ===========================================================================
// 主函数
// ===========================================================================
int main()
{
    int len = 100000000;                       // 1e8
    size_t bytes = sizeof(int) * len;

    int* h_arr = new int[len];
    for (int i = 0; i < len; ++i) h_arr[i] = 1;

    int* d_arr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_arr, bytes));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    int cpu_sum = reduce_cpu(h_arr, len);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double cpu_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 +
                    (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
    printf("CPU sum = %d, time = %.3f ms\n\n", cpu_sum, cpu_ms);

    const int repeat = 20;

    int grid_normal = CEIL_DIV(len, BLOCKSIZE);
    int grid_two    = CEIL_DIV(len, BLOCKSIZE * 2);

    float ms_naive = 0, ms_inter = 0, ms_seq = 0, ms_two = 0, ms_warp = 0;
    int   s_naive = 0, s_inter = 0, s_seq = 0, s_two = 0, s_warp = 0;

    // 1. naive
    {
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * grid_normal));
        ms_naive = bench_kernel(reduce_naive_kernel<BLOCKSIZE>, grid_normal,
                                d_arr, d_out, len, start, stop, repeat);
        s_naive = verify(d_out, grid_normal);
        printf("naive          : %8.3f ms, sum = %d, %s\n",
               ms_naive, s_naive, (s_naive == cpu_sum) ? "success" : "failed");
        cudaFree(d_out);
    }

    // 2. interleaved
    {
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * grid_normal));
        ms_inter = bench_kernel(reduce_interleaved_kernel<BLOCKSIZE>, grid_normal,
                                d_arr, d_out, len, start, stop, repeat);
        s_inter = verify(d_out, grid_normal);
        printf("interleaved    : %8.3f ms, sum = %d, %s\n",
               ms_inter, s_inter, (s_inter == cpu_sum) ? "success" : "failed");
        cudaFree(d_out);
    }

    // 3. sequential
    {
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * grid_normal));
        ms_seq = bench_kernel(reduce_sequential_kernel<BLOCKSIZE>, grid_normal,
                              d_arr, d_out, len, start, stop, repeat);
        s_seq = verify(d_out, grid_normal);
        printf("sequential     : %8.3f ms, sum = %d, %s\n",
               ms_seq, s_seq, (s_seq == cpu_sum) ? "success" : "failed");
        cudaFree(d_out);
    }

    // 4. two-per-thread
    {
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * grid_two));
        ms_two = bench_kernel(reduce_two_per_thread_kernel<BLOCKSIZE>, grid_two,
                              d_arr, d_out, len, start, stop, repeat);
        s_two = verify(d_out, grid_two);
        printf("two-per-thread : %8.3f ms, sum = %d, %s\n",
               ms_two, s_two, (s_two == cpu_sum) ? "success" : "failed");
        cudaFree(d_out);
    }

    // 5. warp_reduce
    {
        int* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * grid_normal));
        ms_warp = bench_kernel(reduce_warp_kernel<BLOCKSIZE>, grid_normal,
                               d_arr, d_out, len, start, stop, repeat);
        s_warp = verify(d_out, grid_normal);
        printf("warp_reduce    : %8.3f ms, sum = %d, %s\n",
               ms_warp, s_warp, (s_warp == cpu_sum) ? "success" : "failed");
        cudaFree(d_out);
    }

    // 总对比
    printf("\n================= Runtime Comparison =================\n");
    printf("Data size        : %d ints (%.2f MB)\n", len, bytes / (1024.0 * 1024.0));
    printf("Block size       : %d\n", BLOCKSIZE);
    printf("Repeat           : %d\n", repeat);
    printf("CPU time         : %10.3f ms\n", cpu_ms);
    printf("------------------------------------------------------\n");
    printf("%-16s %10s %10s %12s\n", "Kernel", "Time(ms)", "Speedup", "GridSize");
    printf("%-16s %10.3f %9.2fx %12d\n", "naive",          ms_naive, cpu_ms / ms_naive, grid_normal);
    printf("%-16s %10.3f %9.2fx %12d\n", "interleaved",    ms_inter, cpu_ms / ms_inter, grid_normal);
    printf("%-16s %10.3f %9.2fx %12d\n", "sequential",     ms_seq,   cpu_ms / ms_seq,   grid_normal);
    printf("%-16s %10.3f %9.2fx %12d\n", "two-per-thread", ms_two,   cpu_ms / ms_two,   grid_two);
    printf("%-16s %10.3f %9.2fx %12d\n", "warp_reduce",    ms_warp,  cpu_ms / ms_warp,  grid_normal);
    printf("======================================================\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_arr));
    delete[] h_arr;
    return 0;
}