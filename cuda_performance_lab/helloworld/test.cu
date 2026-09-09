#include <stdio.h>
void __global__ hello_from_gpu()
{
    const int blockid = blockIdx.x;
    const int threadid = threadIdx.x;
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("hello world from block %d and %d thread, global id %d\n", blockid, threadid, id);
}

int main(void)
{
    printf("hello world from CPU\n");
    hello_from_gpu<<<2,4>>>();
    cudaDeviceSynchronize();

    return 0;
}