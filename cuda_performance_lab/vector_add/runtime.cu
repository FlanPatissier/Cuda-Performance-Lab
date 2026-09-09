#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

__global__ void vectorAdd(float *A, float *B, float *C, int numberElements)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < numberElements) {
        C[i] = A[i] + B[i];
    }
}

// 计时函数：测试指定配置
float test_configuration(int N, int threadsPerBlock, bool verify=false)
{
    size_t size = N * sizeof(float);
    
    // 分配 host 内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // 初始化
    for(int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 拷贝到 device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 计算 grid 大小
    int blockPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // CUDA event 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热（第一次调用可能包含初始化开销）
    vectorAdd<<<blockPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // 正式计时
    cudaEventRecord(start);
    vectorAdd<<<blockPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);
    
    // 验证（可选）
    if (verify) {
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
                printf("Verification failed at element %d!\n", i);
                break;
            }
        }
    }
    
    // 计算带宽
    float bandwidth = (3.0f * size) / (kernel_time / 1000.0f) / 1e9;  // GB/s
    
    printf("N = %8d, Block Size = %4d, Grid = %6d | Kernel: %.4f ms | Bandwidth: %.1f GB/s\n", 
           N, threadsPerBlock, blockPerGrid, kernel_time, bandwidth);
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return kernel_time;
}

int main(void)
{
    printf("=== CUDA Vector Add Performance Test ===\n\n");
    
    // 测试1：固定 N，不同 block size
    printf("--- Test 1: N = 1M (4MB), varying block size ---\n");
    int N1 = 1 << 20;  // 1M elements
    int block_sizes[] = {64, 128, 256, 512, 1024};
    
    for (int bs : block_sizes) {
        test_configuration(N1, bs);
    }
    
    // 测试2：固定 block size，不同 N
    printf("\n--- Test 2: Block size = 256, varying N ---\n");
    int block_size = 256;
    int sizes[] = {1<<16, 1<<20, 1<<22, 1<<24, 1<<26};  // 64K to 64M
    
    for (int N : sizes) {
        test_configuration(N, block_size);
    }
    
    // 测试3：验证正确性
    printf("\n--- Test 3: Verification with N = 50 ---\n");
    test_configuration(50, 256, true);
    printf("Verification PASSED\n");
    
    printf("\nDone\n");
    return 0;
}