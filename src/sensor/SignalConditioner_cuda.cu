#ifdef CUDA_ENABLED

#include "sensor/SignalConditioner.h"
#include "sensor/KinectSensor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

#ifndef CUDA_CHECK_LAST
#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)
#endif

namespace kfusion {
namespace sensor {

// Constants matching SignalConditioner_omp.cpp
namespace {
    constexpr int kHoleFillRadius = 2;
    constexpr int kGuidedRadius = 4; // Reduced from 9: 9x9 kernel ~17M MACs vs 19x19 ~110M
    constexpr int kRgbBilateralRadius = 2;
    constexpr int kDepthMedianRadius = 1;
    constexpr float kEmaJumpResetMeters = 0.05f;
    constexpr float kGuidedSigmaLuma = 0.10f;
    constexpr float kGuidedSigmaDepth = 0.04f;
    constexpr float kRgbSigmaSpatial = 2.0f;
    constexpr float kRgbSigmaRange = 28.0f;

    // Reflect boundary handling to eliminate vertical banding
    __device__ inline int reflectCoordCUDA(int x, int max_val) {
        if (x < 0) return -x - 1;
        if (x >= max_val) return 2 * max_val - x - 1;
        return x;
    }

    // Edge detection for edge-aware filtering
    __device__ inline float computeDepthGradientCUDA(const uint16_t* depth, int x, int y, int w, int h) {
        if (x <= 0 || x >= w - 1 || y <= 0 || y >= h - 1) return 1.0f; // Edge pixel
        
        float center = (float)depth[y * w + x];
        float right = (float)depth[y * w + (x + 1)];
        float left = (float)depth[y * w + (x - 1)];
        float down = (float)depth[(y + 1) * w + x];
        float up = (float)depth[(y - 1) * w + x];
        
        if (center == 0.0f || right == 0.0f || left == 0.0f || down == 0.0f || up == 0.0f) return 1.0f; // Edge pixel
        
        float gx = fabsf(right - left);
        float gy = fabsf(down - up);
        float gradient = sqrtf(gx * gx + gy * gy);
        
        // Normalize gradient to [0,1] range (typical depth range 0-2047)
        return fminf(gradient / 100.0f, 1.0f);
    }

    __device__ inline float rgbLumaGPU(uint8_t r, uint8_t g, uint8_t b) {
        return 0.299f * (float)r + 0.587f * (float)g + 0.114f * (float)b;
    }

    __device__ inline float rawDepthToMetersGPU(uint16_t depth) {
        if (depth == 0 || depth == 2047) return 0.0f;
        return 1.0f / (depth * -0.0030711016f + 3.3309495161f);
    }

    __device__ inline uint16_t metersToRawDepthGPU(float depth_m) {
        if (depth_m <= 0.0f) return 0;
        float raw = (1.0f / depth_m - 3.3309495161f) / -0.0030711016f;
        return (uint16_t)max(1.0f, min(2046.0f, floorf(raw + 0.5f)));
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// FSR 1.0 RCAS — CUDA inline kernel (same algorithm as SuperResolution_cuda.cu)
// ---------------------------------------------------------------------------

__device__ static inline void rcasGetPixelCUDA(const uint8_t* img,
                                                int x, int y, int w, int h,
                                                float out[3]) {
    x = max(0, min(x, w - 1));
    y = max(0, min(y, h - 1));
    const int idx = (y * w + x) * 3;
    out[0] = (float)img[idx + 0] * (1.0f / 255.0f);
    out[1] = (float)img[idx + 1] * (1.0f / 255.0f);
    out[2] = (float)img[idx + 2] * (1.0f / 255.0f);
}

__global__ void applyCASKernel(const uint8_t* __restrict__ d_in,
                               uint8_t* __restrict__ d_out,
                               int width, int height, float peak) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float a[3], b[3], c[3], d[3], e[3];
    rcasGetPixelCUDA(d_in, x,     y - 1, width, height, a);
    rcasGetPixelCUDA(d_in, x - 1, y,     width, height, b);
    rcasGetPixelCUDA(d_in, x,     y,     width, height, c);
    rcasGetPixelCUDA(d_in, x + 1, y,     width, height, d);
    rcasGetPixelCUDA(d_in, x,     y + 1, width, height, e);

    float amp = 1.0f;
    for (int ch = 0; ch < 3; ++ch) {
        float mn     = fminf(a[ch], fminf(b[ch], fminf(c[ch], fminf(d[ch], e[ch]))));
        float mx     = fmaxf(a[ch], fmaxf(b[ch], fmaxf(c[ch], fmaxf(d[ch], e[ch]))));
        float mx_s   = fmaxf(mx, 1e-6f);
        float amp_ch = fminf(mn, 1.0f - mx) / mx_s;
        amp = fminf(amp, amp_ch);
    }

    const float w          = amp * peak;
    const float weight_sum = 1.0f + 4.0f * w;
    const int   out_idx    = (y * width + x) * 3;

    for (int ch = 0; ch < 3; ++ch) {
        float val = (c[ch] + w * (a[ch] + b[ch] + d[ch] + e[ch])) / weight_sum;
        d_out[out_idx + ch] = (uint8_t)(fmaxf(0.0f, fminf(1.0f, val)) * 255.0f);
    }
}

__global__ void bilateralDenoiseRgbKernel(const uint8_t* src, uint8_t* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int center_idx = (y * w + x) * 3;
    float center_r = src[center_idx + 0];
    float center_g = src[center_idx + 1];
    float center_b = src[center_idx + 2];

    float accum_r = 0, accum_g = 0, accum_b = 0, weight_sum = 0;

    for (int dy = -kRgbBilateralRadius; dy <= kRgbBilateralRadius; ++dy) {
        int sy = reflectCoordCUDA(y + dy, h);
        for (int dx = -kRgbBilateralRadius; dx <= kRgbBilateralRadius; ++dx) {
            int sx = reflectCoordCUDA(x + dx, w);
            int sidx = (sy * w + sx) * 3;

            float sr = src[sidx + 0];
            float sg = src[sidx + 1];
            float sb = src[sidx + 2];

            float spatial_dist_sq = (float)(dx * dx + dy * dy);
            float color_dist_sq = (sr - center_r) * (sr - center_r) +
                                  (sg - center_g) * (sg - center_g) +
                                  (sb - center_b) * (sb - center_b);

            float weight = expf(-spatial_dist_sq / (2.0f * kRgbSigmaSpatial * kRgbSigmaSpatial)) *
                           expf(-color_dist_sq / (2.0f * kRgbSigmaRange * kRgbSigmaRange));

            accum_r += weight * sr;
            accum_g += weight * sg;
            accum_b += weight * sb;
            weight_sum += weight;
        }
    }

    dst[center_idx + 0] = (uint8_t)(accum_r / weight_sum);
    dst[center_idx + 1] = (uint8_t)(accum_g / weight_sum);
    dst[center_idx + 2] = (uint8_t)(accum_b / weight_sum);
}

__global__ void medianBlur3x3Kernel(const uint8_t* src, uint8_t* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    for (int c = 0; c < 3; ++c) {
        uint8_t window[9];
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            int sy = reflectCoordCUDA(y + dy, h);
            for (int dx = -1; dx <= 1; ++dx) {
                int sx = reflectCoordCUDA(x + dx, w);
                window[count++] = src[(sy * w + sx) * 3 + c];
            }
        }
        // Simple bubble sort for median of 9
        for (int i = 0; i < 5; ++i) {
            for (int j = i + 1; j < 9; ++j) {
                if (window[i] > window[j]) {
                    uint8_t tmp = window[i];
                    window[i] = window[j];
                    window[j] = tmp;
                }
            }
        }
        dst[(y * w + x) * 3 + c] = window[4];
    }
}

__global__ void computeLumaKernel(const uint8_t* rgb, float* luma, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    luma[idx] = rgbLumaGPU(rgb[idx * 3 + 0], rgb[idx * 3 + 1], rgb[idx * 3 + 2]) / 255.0f;
}

__global__ void denoiseDepthSpatialKernel(const uint16_t* src, uint16_t* dst, int w, int h, float min_d, float max_d) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    float center_d = rawDepthToMetersGPU(src[idx]);
    if (center_d < min_d || center_d > max_d) {
        dst[idx] = src[idx];
        return;
    }

    float window[9];
    int count = 0;
    for (int dy = -kDepthMedianRadius; dy <= kDepthMedianRadius; ++dy) {
        int sy = reflectCoordCUDA(y + dy, h);
        for (int dx = -kDepthMedianRadius; dx <= kDepthMedianRadius; ++dx) {
            int sx = reflectCoordCUDA(x + dx, w);
            float d = rawDepthToMetersGPU(src[sy * w + sx]);
            if (d >= min_d && d <= max_d) {
                window[count++] = d;
            }
        }
    }

    if (count >= 3) {
        for (int i = 0; i <= count / 2; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (window[i] > window[j]) {
                    float tmp = window[i];
                    window[i] = window[j];
                    window[j] = tmp;
                }
            }
        }
        dst[idx] = metersToRawDepthGPU(window[count / 2]);
    } else {
        dst[idx] = src[idx];
    }
}

__global__ void applyDepthEmaKernel(uint16_t* depth, float* ema_buf, int w, int h, float min_d, float max_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    float d = rawDepthToMetersGPU(depth[idx]);
    if (d < min_d || d > max_d || isnan(d) || isinf(d)) {
        ema_buf[idx] = 0.0f;
        return;
    }

    float previous = ema_buf[idx];
    float delta = fabsf(d - previous);
    
    // Improved EMA stability: check for temporal consistency with neighbors
    bool reset_ema = (isnan(previous) || isinf(previous) || previous <= 0.0f || delta > kEmaJumpResetMeters);
    
    // Additional validation: if delta is suspiciously large, check neighboring pixels
    if (!reset_ema && delta > kEmaJumpResetMeters * 0.5f) {
        int x = idx % w;
        int y = idx / w;
        float neighbor_sum = 0.0f;
        int neighbor_count = 0;
        
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = reflectCoordCUDA(x + dx, w);
                int ny = reflectCoordCUDA(y + dy, h);
                int ni = ny * w + nx;
                float nd = rawDepthToMetersGPU(depth[ni]);
                if (nd >= min_d && nd <= max_d) {
                    neighbor_sum += nd;
                    neighbor_count++;
                }
            }
        }
        
        if (neighbor_count > 0) {
            float neighbor_avg = neighbor_sum / neighbor_count;
            float neighbor_delta = fabsf(d - neighbor_avg);
            // If current depth is very different from neighbors, reset EMA
            if (neighbor_delta > kEmaJumpResetMeters) {
                reset_ema = true;
            }
        }
    }
    
    float filtered = reset_ema ? d : (0.7f * d + 0.3f * previous);

    ema_buf[idx] = filtered;
    depth[idx] = metersToRawDepthGPU(filtered);
}

__global__ void fillDepthHolesKernel(const uint16_t* src, uint16_t* dst, int w, int h, float min_d, float max_d) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    if (src[idx] != 0 && src[idx] != 2047) {
        dst[idx] = src[idx];
        return;
    }

    // Edge-aware hole filling: use weighted average instead of nearest neighbor
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    int valid_neighbors = 0;

    for (int dy = -kHoleFillRadius; dy <= kHoleFillRadius; ++dy) {
        int sy = reflectCoordCUDA(y + dy, h);
        for (int dx = -kHoleFillRadius; dx <= kHoleFillRadius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int sx = reflectCoordCUDA(x + dx, w);

            float candidate = rawDepthToMetersGPU(src[sy * w + sx]);
            if (candidate >= min_d && candidate <= max_d) {
                int dist_sq = dx * dx + dy * dy;
                float weight = 1.0f / (dist_sq + 1.0f); // Inverse distance weighting
                weighted_sum += weight * candidate;
                weight_sum += weight;
                valid_neighbors++;
            }
        }
    }

    if (valid_neighbors > 0 && weight_sum > 1e-6f) {
        dst[idx] = metersToRawDepthGPU(weighted_sum / weight_sum);
    } else {
        dst[idx] = 0;
    }
}

__global__ void guidedDepthFilterKernel(const uint16_t* depth, uint16_t* dst, const float* luma, int w, int h, float min_d, float max_d) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    float center_d = rawDepthToMetersGPU(depth[idx]);
    if (center_d <= 0.01f) {
        dst[idx] = 0;
        return;
    }
    if (center_d < min_d || center_d > max_d) {
        dst[idx] = depth[idx];
        return;
    }

    float center_l = luma[idx];
    
    // Edge-aware filtering: compute depth gradient to reduce filter strength near edges
    float edge_strength = computeDepthGradientCUDA(depth, x, y, w, h);

    float sum_w = 0.0f;
    float sum_d = 0.0f;

    for (int dy = -kGuidedRadius; dy <= kGuidedRadius; ++dy) {
        int sy = reflectCoordCUDA(y + dy, h);
        for (int dx = -kGuidedRadius; dx <= kGuidedRadius; ++dx) {
            int sx = reflectCoordCUDA(x + dx, w);

            int nidx = sy * w + sx;
            float nd = rawDepthToMetersGPU(depth[nidx]);
            if (nd < min_d || nd > max_d) continue;

            float spatial_dist_sq = (float)(dx * dx + dy * dy);
            float l_delta = luma[nidx] - center_l;
            float d_delta = nd - center_d;

            float spatial_weight = expf(-spatial_dist_sq / (2.0f * kGuidedRadius * kGuidedRadius));
            float luma_weight = expf(-(l_delta * l_delta) / (2.0f * kGuidedSigmaLuma * kGuidedSigmaLuma + 0.01f));
            float depth_weight = expf(-(d_delta * d_delta) / (2.0f * kGuidedSigmaDepth * kGuidedSigmaDepth + 0.01f));
            
            // Edge-aware: reduce filter strength near edges to prevent blurring
            float edge_factor = 1.0f - edge_strength * 0.5f; // Reduce by up to 50% near edges
            float weight = spatial_weight * luma_weight * depth_weight * edge_factor;

            sum_w += weight;
            sum_d += weight * nd;
        }
    }

    if (sum_w > 1e-6f) {
        dst[idx] = metersToRawDepthGPU(sum_d / sum_w);
    } else {
        dst[idx] = depth[idx];
    }
}

__global__ void rawToMetersKernel(const uint16_t* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = rawDepthToMetersGPU(src[idx]);
}

// ---------------------------------------------------------------------------
// SignalConditioner::processCuda
// ---------------------------------------------------------------------------

bool SignalConditioner::processCuda(RawFrame& raw,
                                    cudaStream_t stream,
                                    float min_depth_m,
                                    float max_depth_m) {
    int w = RGB_WIDTH;
    int h = RGB_HEIGHT;
    size_t n = w * h;

    // Allocate resources on first use
    if (!d_rgb_in_) {
        d_rgb_in_ = utils::make_cuda_unique<uint8_t>(n * 3);
        d_rgb_out_ = utils::make_cuda_unique<uint8_t>(n * 3);
        d_depth_in_ = utils::make_cuda_unique<uint16_t>(n);
        d_depth_out_ = utils::make_cuda_unique<uint16_t>(n);
        d_depth_meters_ = utils::make_cuda_unique<float>(n);
        d_ema_buf_m_ = utils::make_cuda_unique<float>(n);
        d_guidance_luma_ = utils::make_cuda_unique<float>(n);
        cudaMemsetAsync(d_ema_buf_m_.get(), 0, n * sizeof(float), stream);
    }

    // Upload
    cudaMemcpyAsync(d_rgb_in_.get(), raw.rgb.data(), n * 3, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_depth_in_.get(), raw.depth.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // 1. Preprocess RGB: Bilateral + Median
    bilateralDenoiseRgbKernel<<<grid, block, 0, stream>>>(d_rgb_in_.get(), d_rgb_out_.get(), w, h);
    CUDA_CHECK_LAST();
    medianBlur3x3Kernel<<<grid, block, 0, stream>>>(d_rgb_out_.get(), d_rgb_in_.get(), w, h);
    CUDA_CHECK_LAST();

    // 2. Super Resolution Guidance (CAS)
    // Reduced sharpness from 0.85 to 0.5 to avoid over-sharpening artifacts and quality degradation
    float sharpness = 0.5f;
    float t = std::max(0.0f, std::min(sharpness, 1.0f));
    float peak = -1.0f / ((1.0f - t) * 8.0f + t * 5.0f);
    applyCASKernel<<<grid, block, 0, stream>>>(d_rgb_in_.get(), d_rgb_out_.get(), w, h, peak);
    CUDA_CHECK_LAST();
    
    // Compute Luma for guided filter
    computeLumaKernel<<<grid, block, 0, stream>>>(d_rgb_out_.get(), d_guidance_luma_.get(), w, h);
    CUDA_CHECK_LAST();

    // 3. Depth pipeline
    denoiseDepthSpatialKernel<<<grid, block, 0, stream>>>(d_depth_in_.get(), d_depth_out_.get(), w, h, min_depth_m, max_depth_m);
    CUDA_CHECK_LAST();
    
    int total_threads = n;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    if (!minimal_depth_) {   // minimal: the spatial median above is the whole depth chain
        fillDepthHolesKernel<<<grid, block, 0, stream>>>(d_depth_out_.get(), d_depth_in_.get(), w, h, min_depth_m, max_depth_m);
        CUDA_CHECK_LAST();

        // 4. Guided Depth Filter
        guidedDepthFilterKernel<<<grid, block, 0, stream>>>(d_depth_in_.get(), d_depth_out_.get(), d_guidance_luma_.get(), w, h, min_depth_m, max_depth_m);
        CUDA_CHECK_LAST();

        // 5. Temporal Stabilization (EMA)
        applyDepthEmaKernel<<<grid_size, block_size, 0, stream>>>(d_depth_out_.get(), d_ema_buf_m_.get(), w, h, min_depth_m, max_depth_m);
        CUDA_CHECK_LAST();
    }

    // 5. Final meters conversion for TSDF integration (Zero-copy GPU path)
    rawToMetersKernel<<<grid_size, block_size, 0, stream>>>(d_depth_out_.get(), d_depth_meters_.get(), n);
    CUDA_CHECK_LAST();

    // Download (Only for UI preview or CPU-mode compatibility)
    cudaMemcpyAsync(raw.rgb.data(), d_rgb_out_.get(), n * 3, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(raw.depth.data(), d_depth_out_.get(), n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    return true;
}

} // namespace sensor
} // namespace kfusion

#endif
