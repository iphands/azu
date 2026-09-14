#pragma once

#include <vector>
#include <cstdint>

namespace kfusion {
namespace sensor {
namespace sr {

/**
 * @brief Applies CPU-based Edge Adaptive Spatial Upsampling (AMD FSR 1.0 EASU) to upscale RGB image.
 * 
 * @param src Source RGB byte array
 * @param dst Destination RGB byte array (will be resized)
 * @param src_width Source image width
 * @param src_height Source image height
 * @param scale Scale factor (2, 3, or 4)
 */
void applyEASU_CPU(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                   int src_width, int src_height, int scale);

/**
 * @brief Applies CPU-based Contrast Adaptive Sharpening (AMD FSR 1.0 CAS math) in-place to an RGB image.
 */
void applyCAS_CPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness = 0.85f);

// KIN-FORK(sr): upstream declares applyEASU_GPU for CUDA too, but
// SuperResolution_cuda.cu only implements applyCAS_GPU (easuKernel lives
// solely in the HIP twin) -> undefined reference at link. CUDA falls back to
// applyEASU_CPU; gate the declaration HIP-only so direct calls fail at
// compile, not link.
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
/**
 * @brief Applies GPU-accelerated Contrast Adaptive Sharpening (AMD FSR 1.0 CAS math) in-place.
 */
void applyCAS_GPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness = 0.85f);
#endif
#ifdef HIP_ENABLED
/**
 * @brief Applies GPU-accelerated Edge Adaptive Spatial Upsampling (AMD FSR 1.0 EASU).
 */
void applyEASU_GPU(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                   int src_width, int src_height, int scale);
#endif

/**
 * @brief Master wrapper for EASU. Dispatches to GPU if available, otherwise CPU.
 */
inline void applyEASU(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                     int src_width, int src_height, int scale) {
#ifdef HIP_ENABLED
    applyEASU_GPU(src, dst, src_width, src_height, scale);
#else
    applyEASU_CPU(src, dst, src_width, src_height, scale);
#endif
}

/**
 * @brief Master wrapper. Dispatches to GPU if CUDA is enabled and available, otherwise CPU.
 * 
 * @param rgb Flat RGB byte array of size (width * height * 3)
 * @param width Image width
 * @param height Image height
 * @param sharpness Range [0.0, 1.0]. 0.0 is softest, 1.0 is sharpest. Default is 0.85.
 */
inline void applyCAS(std::vector<uint8_t>& rgb, int width, int height, float sharpness = 0.85f) {
#ifdef CUDA_ENABLED
    applyCAS_GPU(rgb, width, height, sharpness);
#elif defined(HIP_ENABLED)
    applyCAS_GPU(rgb, width, height, sharpness);
#else
    applyCAS_CPU(rgb, width, height, sharpness);
#endif
}

} // namespace sr
} // namespace sensor
} // namespace kfusion
