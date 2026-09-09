#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if(error_code != cudaSuccess)
    {
        printf("Cuda error:\r\ncode = %d, name = %s, description = %s\r\nfile =%s, line = %d\r\n",
        error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

int main(void)
{
    float *fpHost_A;
    fpHost_A = (float *)malloc(4);
    memset(fpHost_A, 0, 4);
    
    float *fpDevice_A;
    cudaError_t error = ErrorCheck(cudaMalloc((float**)&fpDevice_A,4), __FILE__, __LINE__);
    cudaMemset(fpDevice_A, 0, 4);

    ErrorCheck(cudaMemcpy(fpDevice_A, fpHost_A, 4, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    free(fpHost_A);
    ErrorCheck(cudaFree(fpDevice_A), __FILE__, __LINE__);

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
    return 0;
}