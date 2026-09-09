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