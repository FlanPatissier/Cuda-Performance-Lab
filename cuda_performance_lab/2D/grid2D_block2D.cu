#include <stdio.h>
#include "../errorCheck/errorCheckFunction.cuh"

__global__ void vectorAdd(float *A, float *B, float *C, const int nx, const int ny)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int idx = iy * nx + ix;
    if(ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    // calculate the allocated size
    int nx = 15;
    int ny = 9;
    int numberElements = nx * ny;
    size_t size = numberElements * sizeof(float);
    cudaError_t err = cudaSuccess;

    // allocate size in host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // should set them to be 0 first
    if(h_A!=NULL && h_B!=NULL && h_C!=NULL)
    {
        for(int i = 0; i < numberElements; i++)
        {
            h_A[i] = i;
            h_B[i] = i + 1;
        }
        memset(h_C, 0, size);
    } else {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }



    // device allocate size
    float *d_A,*d_B, *d_C;

    err        = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err        = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err        = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //transfer from host to device
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to devive (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to devive (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from host to devive (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // set the threads and block and transfer
    dim3 block(16, 8);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    vectorAdd<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    printf("CUDA kernel launch with %d blocks of %d threads\n", nx, ny);

    ErrorCheck(cudaGetLastError(), __FILE__, __LINE__ );
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__ );


    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numberElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        } else {
        printf("idx = %2d\tA=%.2f\tB=%.2f\tresult=%.2f\n", i+1, h_A[i], h_B[i], h_C[i]);
        }
    }

    printf("Test PASSED\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;



}