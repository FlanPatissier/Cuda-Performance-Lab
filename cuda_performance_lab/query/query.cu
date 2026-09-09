#include <stdio.h>
#include "../errorCheck/errorCheckFunction.cuh"

int main(void)
{
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

    printf("Device id:             %d\n", device_id);
    printf("Device name:           %s\n", prop.name);
    printf("Compute capability:            %d.%d\n", prop.major, prop.minor);
    printf("Amount of global memory:            %g GB\n", prop.totalGlobalMem / (1024 * 1024.0 *1024));
    printf("Amount of constant memory:          %g KB\n", prop.totalConstMem / 1024.0);
    printf("Maximum grid size:                  %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                  %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

}