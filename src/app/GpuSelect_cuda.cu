#ifdef CUDA_ENABLED

#include "app/GpuSelect.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace kfusion {
namespace app {

namespace {
std::string mb(size_t bytes) { return std::to_string(bytes >> 20) + " MB"; }
}

GpuChoice selectCudaDevice(size_t need_bytes) {
    GpuChoice choice;
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        choice.description = std::string("no CUDA device (") +
                             (err != cudaSuccess ? cudaGetErrorString(err) : "count 0") + ")";
        (void)cudaGetLastError();
        return choice;
    }

    int forced = -1;
    if (const char* env = std::getenv("AZU_CUDA_DEVICE")) forced = std::atoi(env);

    size_t best_free = 0;
    std::string rejected;
    for (int d = 0; d < count; ++d) {
        if (forced >= 0 && d != forced) continue;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        size_t free_b = 0, total_b = 0;
        if (cudaSetDevice(d) != cudaSuccess || cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        const int cc = prop.major * 10 + prop.minor;
        const std::string tag = "dev" + std::to_string(d) + " " + prop.name;
        if (cc < 89) {
            rejected += tag + ": sm_" + std::to_string(cc) + " < sm_89; ";
            continue;
        }
        if (free_b < need_bytes) {
            rejected += tag + ": " + mb(free_b) + " free < " + mb(need_bytes) + " needed; ";
            continue;
        }
        if (free_b > best_free) {
            best_free = free_b;
            choice.device = d;
            choice.description = tag + " (" + mb(free_b) + " free, " + mb(need_bytes) + " needed)";
        }
    }

    if (choice.device < 0) {
        choice.description = forced >= 0 && forced >= count
            ? "AZU_CUDA_DEVICE=" + std::to_string(forced) + " does not exist"
            : "no CUDA device fits: " + rejected;
        return choice;
    }
    if (cudaSetDevice(choice.device) != cudaSuccess) {
        choice.description = "cudaSetDevice(" + std::to_string(choice.device) + ") failed";
        choice.device = -1;
    }
    return choice;
}

} // namespace app
} // namespace kfusion

#endif // CUDA_ENABLED
