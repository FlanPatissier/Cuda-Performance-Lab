#include <stdio.h>
#include "../errorCheck/errorCheckFunction.cuh"

#include <cuda_runtime.h>

// 根据 SM 版本获取每个 SM 的 CUDA Core 数
static int convertSMVerToCores(int major, int minor)
{
    struct SMToCores {
        int sm;     // 例如 0x86 表示 8.6
        int cores;  // 每个 SM 的 CUDA cores
    };

    // 常见 NVIDIA 架构映射（含主流数据中心/消费级）
    static const SMToCores map[] = {
        {0x30, 192}, // Kepler
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},

        {0x50, 128}, // Maxwell
        {0x52, 128},
        {0x53, 128},

        {0x60,  64}, // Pascal
        {0x61, 128},
        {0x62, 128},

        {0x70,  64}, // Volta
        {0x72,  64},

        {0x75,  64}, // Turing

        {0x80,  64}, // Ampere (A100)
        {0x86, 128}, // Ampere (RTX30 等)
        {0x87, 128},
        {0x89, 128}, // Ada

        {0x90, 128}, // Hopper (H100)
        {0x00,  -1}
    };

    int sm = (major << 4) + minor;
    for (int i = 0; map[i].sm != 0x00; ++i) {
        if (map[i].sm == sm) return map[i].cores;
    }

    // 未知新架构时给一个保守默认值（常见为 128）
    return 128;
}

// 你要的函数：根据 prop 计算总 SP（CUDA Core）数量
int getSPScores(const cudaDeviceProp& prop)
{
    int coresPerSM = convertSMVerToCores(prop.major, prop.minor);
    return coresPerSM * prop.multiProcessorCount;
}

int main(void)
{
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int spCores = getSPScores(prop);

    printf("GPU: %s\n", prop.name);
    printf("SM count: %d, CC: %d.%d, Estimated CUDA cores: %d\n",
       prop.multiProcessorCount, prop.major, prop.minor, spCores);
}