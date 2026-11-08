// transpose_nvidia.cu
// 基于 NVIDIA CUDA Samples 的 transpose 示例改写
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

const int TILE_DIM   = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS   = 100;

// 校验并打印有效带宽（GB/s）
void postprocess(const float *ref, const float *res, int n, float ms)
{
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (res[i] != ref[i]) {
            printf("  mismatch at %d: got %f, expected %f\n", i, res[i], ref[i]);
            printf("%25s\n", "*** FAILED ***");
            passed = false;
            break;
        }
    }
    if (passed) {
        // 2 * n * sizeof(float)：读 + 写
        printf("%20.2f\n", 2.0 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
    }
}

// ---------------------------------------------------------------------------
// 1. 纯拷贝：作为“最佳有效带宽”的参考
// ---------------------------------------------------------------------------
__global__ void copy(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

// ---------------------------------------------------------------------------
// 2. 经共享内存的拷贝：演示共享内存本身的开销
// ---------------------------------------------------------------------------
__global__ void copySharedMem(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM * TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
}

// ---------------------------------------------------------------------------
// 3. 朴素转置：读合并，写不合并
// ---------------------------------------------------------------------------
__global__ void transposeNaive(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
}

// ---------------------------------------------------------------------------
// 4. 合并转置：共享内存中转，读写都合并；但 tile[32][32] 有 bank conflict
// ---------------------------------------------------------------------------
__global__ void transposeCoalesced(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;   // 转置后的块偏移
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------------------------------------------------------------------------
// 5. 无 bank conflict 转置：tile[32][33]，多一列 padding
// ---------------------------------------------------------------------------
__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    const int nx = (argc > 1) ? atoi(argv[1]) : 1024;
    const int ny = nx;
    const int mem_size = nx * ny * sizeof(float);

    if (nx % TILE_DIM || ny % TILE_DIM) {
        printf("nx and ny must be a multiple of TILE_DIM (%d)\n", TILE_DIM);
        return 1;
    }
    if (TILE_DIM % BLOCK_ROWS) {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        return 1;
    }

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    int devId = 0;
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, devId));
    printf("\nDevice : %s\n", prop.name);
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
           nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    checkCuda(cudaSetDevice(devId));

    float *h_idata = (float*)malloc(mem_size);
    float *h_cdata = (float*)malloc(mem_size);
    float *h_tdata = (float*)malloc(mem_size);
    float *gold    = (float*)malloc(mem_size);

    float *d_idata, *d_cdata, *d_tdata;
    checkCuda(cudaMalloc(&d_idata, mem_size));
    checkCuda(cudaMalloc(&d_cdata, mem_size));
    checkCuda(cudaMalloc(&d_tdata, mem_size));

    // 初始化输入
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            h_idata[j * nx + i] = (float)(j * nx + i);

    // 正确的转置结果
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            gold[j * nx + i] = h_idata[i * nx + j];

    checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    float ms;

    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

    // ---- copy ----
    printf("%25s", "copy");
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(h_idata, h_cdata, nx * ny, ms);

    // ---- copySharedMem ----
    printf("%25s", "shared memory copy");
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(h_idata, h_cdata, nx * ny, ms);

    // ---- transposeNaive ----
    printf("%25s", "naive transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);

    // ---- transposeCoalesced ----
    printf("%25s", "coalesced transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);

    // ---- transposeNoBankConflicts ----
    printf("%25s", "conflict-free transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx * ny, ms);

    // cleanup
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    checkCuda(cudaFree(d_tdata));
    checkCuda(cudaFree(d_cdata));
    checkCuda(cudaFree(d_idata));
    free(h_idata);
    free(h_tdata);
    free(h_cdata);
    free(gold);

    return 0;
}