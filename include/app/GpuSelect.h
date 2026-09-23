#pragma once

#include <cstddef>
#include <string>

namespace kfusion {
namespace app {

// Device memory the CUDA pipeline allocates up front for a dense volume of
// `resolution`^3: VoxelGPU volume (32 B/voxel), marching-cubes per-voxel
// counts/offsets (8 B/voxel) and its 2M-triangle output buffers, three model
// frames, the ICP pyramid and a margin for the conditioner and driver.
inline size_t cudaBudgetBytes(int resolution) {
    const size_t voxels = static_cast<size_t>(resolution) * resolution * resolution;
    const size_t mc_out = static_cast<size_t>(2000000) * 3 * (12 + 12 + 3);
    const size_t models = static_cast<size_t>(3) * 640 * 480 * (12 + 12 + 3);
    return voxels * (32 + 8) + mc_out + models + (static_cast<size_t>(128) << 20);
}

struct GpuChoice {
    int         device = -1;   // -1: no usable device, run on CPU
    std::string description;   // what was chosen, or why nothing was
};

#ifdef CUDA_ENABLED
// Pick the CUDA device to run on and make it current on the calling thread.
// AZU_CUDA_DEVICE=<n> forces device n. Otherwise the device with the most free
// memory among those that can run this binary (compute capability >= 8.9, the
// oldest architecture it is built for) and have at least `need_bytes` free.
// CUDA_VISIBLE_DEVICES is honoured by the runtime as usual.
GpuChoice selectCudaDevice(size_t need_bytes);
#endif

} // namespace app
} // namespace kfusion
