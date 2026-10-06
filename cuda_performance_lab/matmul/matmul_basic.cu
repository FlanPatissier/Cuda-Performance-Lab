// matmul.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

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

// A: M x K
// B: K x N
// C: M x N
__global__ void matmul_kernel(
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

int main()
{
    int M = 256;
    int N = 320;
    int K = 128;

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
    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,
        bytesA,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B,
        bytesB,
        cudaMemcpyHostToDevice));

    // 6. 设置 kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(
        CEIL_DIV(N, blockDim.x),
        CEIL_DIV(M, blockDim.y));

    // 7. 执行矩阵乘法
    matmul_kernel<<<gridDim, blockDim>>>(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. Device -> Host
    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        bytesC,
        cudaMemcpyDeviceToHost));

    // 9. CPU 计算参考结果
    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);

    // 10. 比较结果
    float maxError = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        float error = fabsf(h_C[i] - h_C_ref[i]);

        if (error > maxError)
        {
            maxError = error;
        }
    }

    printf("A: %d x %d\n", M, K);
    printf("B: %d x %d\n", K, N);
    printf("C: %d x %d\n", M, N);
    printf("Maximum error: %e\n", maxError);

    if (maxError < 1e-3f)
    {
        printf("Result: PASS\n");
    }
    else
    {
        printf("Result: FAIL\n");
    }

    // 11. 释放 GPU 内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // 12. 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return 0;
}