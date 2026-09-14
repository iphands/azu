#include "sensor/SignalConditioner.h"

#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "sensor/SuperResolution.h"
#include "utils/Logger.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <fstream>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include <hip/hip_runtime.h>
#endif

namespace kfusion {
namespace sensor {

namespace {

constexpr int kClaheTileSize = 8;
constexpr int kHoleFillRadius = 2;
// Reduced from 9 to 4: radius-9 = 19x19 kernel = ~110M MACs/frame on 5650u iGPU.
// Radius-4 = 9x9 kernel = ~17M MACs/frame with near-identical denoising quality.
constexpr int kGuidedRadius = 4;
constexpr int kRgbBilateralRadius = 2;
constexpr int kDepthMedianRadius = 1;
constexpr float kEmaJumpResetMeters = 0.05f;
constexpr float kGuidedSigmaLuma = 0.10f;
constexpr float kGuidedSigmaDepth = 0.04f;
constexpr float kRgbSigmaSpatial = 2.0f;
constexpr float kRgbSigmaRange = 28.0f;

// Logging infrastructure
struct FilterStats {
    int bilateral_filtered = 0;
    int median_filtered = 0;
    int hole_filled = 0;
    int guided_filtered = 0;
    int ema_reset = 0;
    int ema_filtered = 0;
    int boundary_clamps = 0;
    int edge_pixels = 0;
    float max_depth_delta = 0.0f;
    float avg_depth_delta = 0.0f;
};

FilterStats g_stats;
bool g_logging_enabled = false;
std::ofstream g_log_file;

void initLogging() {
    const char* log_env = std::getenv("KFUSION_LOG");
    if (log_env) {
        std::string env_str(log_env);
        for (auto& c : env_str) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (env_str == "debug" || env_str == "1" || env_str == "verbose") {
            g_logging_enabled = true;
            g_log_file.open("signal_conditioner_log.csv");
            if (g_log_file.is_open()) {
                g_log_file << "frame,bilateral_filtered,median_filtered,hole_filled,guided_filtered,ema_reset,ema_filtered,boundary_clamps,edge_pixels,max_depth_delta,avg_depth_delta\n";
            }
        }
    }
}

void logFrameStats(int frame_id) {
    if (!g_logging_enabled || !g_log_file.is_open()) return;
    g_log_file << frame_id << ","
               << g_stats.bilateral_filtered << ","
               << g_stats.median_filtered << ","
               << g_stats.hole_filled << ","
               << g_stats.guided_filtered << ","
               << g_stats.ema_reset << ","
               << g_stats.ema_filtered << ","
               << g_stats.boundary_clamps << ","
               << g_stats.edge_pixels << ","
               << g_stats.max_depth_delta << ","
               << g_stats.avg_depth_delta << "\n";
    g_log_file.flush();
    // Reset stats for next frame
    g_stats = FilterStats{};
}

// Reflect boundary handling to eliminate vertical banding
inline int reflectCoord(int x, int max_val) {
    if (x < 0) return -x - 1;
    if (x >= max_val) return 2 * max_val - x - 1;
    return x;
}

inline uint8_t clampToByte(float v) {
    return static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
}

// Edge detection for edge-aware filtering
inline float computeDepthGradient(const std::vector<uint16_t>& depth, int x, int y, int w, int h) {
    if (x <= 0 || x >= w - 1 || y <= 0 || y >= h - 1) return 1.0f; // Edge pixel
    
    float center = static_cast<float>(depth[y * w + x]);
    float right = static_cast<float>(depth[y * w + (x + 1)]);
    float left = static_cast<float>(depth[y * w + (x - 1)]);
    float down = static_cast<float>(depth[(y + 1) * w + x]);
    float up = static_cast<float>(depth[(y - 1) * w + x]);
    
    if (center == 0 || right == 0 || left == 0 || down == 0 || up == 0) return 1.0f; // Edge pixel
    
    float gx = std::abs(right - left);
    float gy = std::abs(down - up);
    float gradient = std::sqrt(gx * gx + gy * gy);
    
    // Normalize gradient to [0,1] range (typical depth range 0-2047)
    return std::min(gradient / 100.0f, 1.0f);
}

inline float rgbLuma(const uint8_t* rgb) {
    return 0.299f * static_cast<float>(rgb[0]) +
           0.587f * static_cast<float>(rgb[1]) +
           0.114f * static_cast<float>(rgb[2]);
}

inline uint16_t metersToRawDepth(float depth_m) {
    if (depth_m <= 0.0f) return 0;
    const float raw = (1.0f / depth_m - 3.3309495161f) / -0.0030711016f;
    // Use explicit long cast to avoid signed/unsigned narrowing on MSVC where
    // long is 32-bit: std::lround returns long, clamp literals must match.
    const long clamped = std::clamp(static_cast<long>(std::lround(raw)), 1L, 2046L);
    return static_cast<uint16_t>(clamped);
}

inline bool isValidDepthMeters(float d, float min_depth_m, float max_depth_m) {
    return d >= min_depth_m && d <= max_depth_m;
}

void medianBlur3x3(std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> scratch(rgb.size(), 0);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            for (int c = 0; c < 3; ++c) {
                uint8_t window[9];
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int sy = reflectCoord(y + dy, FRAME_H);
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = reflectCoord(x + dx, FRAME_W);
                        window[count++] = rgb[(sy * FRAME_W + sx) * 3 + c];
                    }
                }
                std::nth_element(window, window + 4, window + count);
                scratch[(y * FRAME_W + x) * 3 + c] = window[4];
            }
        }
    }

    rgb.swap(scratch);
    if (g_logging_enabled) {
        #pragma omp atomic
        g_stats.median_filtered += FRAME_W * FRAME_H;
    }
}

void bilateralDenoiseRgb(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int center_idx = (y * FRAME_W + x) * 3;
            const float center_rgb[3] = {
                static_cast<float>(src[center_idx + 0]),
                static_cast<float>(src[center_idx + 1]),
                static_cast<float>(src[center_idx + 2])
            };

            float accum[3] = {0.0f, 0.0f, 0.0f};
            float weight_sum = 0.0f;

            for (int dy = -kRgbBilateralRadius; dy <= kRgbBilateralRadius; ++dy) {
                const int sy = reflectCoord(y + dy, FRAME_H);
                for (int dx = -kRgbBilateralRadius; dx <= kRgbBilateralRadius; ++dx) {
                    const int sx = reflectCoord(x + dx, FRAME_W);
                    const int sample_idx = (sy * FRAME_W + sx) * 3;
                    const float spatial_dist_sq = static_cast<float>(dx * dx + dy * dy);

                    float color_dist_sq = 0.0f;
                    for (int c = 0; c < 3; ++c) {
                        const float delta = static_cast<float>(src[sample_idx + c]) - center_rgb[c];
                        color_dist_sq += delta * delta;
                    }

                    const float spatial_weight = std::exp(-spatial_dist_sq / (2.0f * kRgbSigmaSpatial * kRgbSigmaSpatial));
                    const float range_weight = std::exp(-color_dist_sq / (2.0f * kRgbSigmaRange * kRgbSigmaRange));
                    const float weight = spatial_weight * range_weight;

                    for (int c = 0; c < 3; ++c) {
                        accum[c] += weight * static_cast<float>(src[sample_idx + c]);
                    }
                    weight_sum += weight;
                }
            }

            for (int c = 0; c < 3; ++c) {
                dst[center_idx + c] = (weight_sum > 1e-6f)
                    ? clampToByte(accum[c] / weight_sum)
                    : src[center_idx + c];
            }
        }
    }
    if (g_logging_enabled) {
        #pragma omp atomic
        g_stats.bilateral_filtered += FRAME_W * FRAME_H;
    }
}

} // namespace

SignalConditioner::SignalConditioner()
    : ema_buf_m_(FRAME_W * FRAME_H, 0.0f),
      sr_rgb_(FRAME_W * FRAME_H * 3, 0),
      sr_rgb_upscaled_(FRAME_W * FRAME_H * 3, 0),
      rgb_scratch_(FRAME_W * FRAME_H * 3, 0),
      guidance_luma_(FRAME_W * FRAME_H, 0.0f),
      depth_scratch_(FRAME_W * FRAME_H, 0) {}

void SignalConditioner::reset() {
    resetEMA();
    std::fill(sr_rgb_.begin(), sr_rgb_.end(), 0);
    std::fill(sr_rgb_upscaled_.begin(), sr_rgb_upscaled_.end(), 0);
    std::fill(rgb_scratch_.begin(), rgb_scratch_.end(), 0);
    std::fill(guidance_luma_.begin(), guidance_luma_.end(), 0.0f);
    std::fill(depth_scratch_.begin(), depth_scratch_.end(), 0);

#ifdef CUDA_ENABLED
    if (d_rgb_in_) cudaMemset(d_rgb_in_.get(), 0, sr_rgb_.size());
    if (d_rgb_out_) cudaMemset(d_rgb_out_.get(), 0, sr_rgb_.size());
    if (d_guidance_luma_) cudaMemset(d_guidance_luma_.get(), 0, guidance_luma_.size() * sizeof(float));
    if (d_depth_in_) cudaMemset(d_depth_in_.get(), 0, depth_scratch_.size() * sizeof(uint16_t));
    if (d_depth_out_) cudaMemset(d_depth_out_.get(), 0, depth_scratch_.size() * sizeof(uint16_t));
#elif defined(HIP_ENABLED)
    if (d_rgb_in_) (void)hipMemset(d_rgb_in_.get(), 0, sr_rgb_.size());
    if (d_rgb_out_) (void)hipMemset(d_rgb_out_.get(), 0, sr_rgb_.size());
    if (d_guidance_luma_) (void)hipMemset(d_guidance_luma_.get(), 0, guidance_luma_.size() * sizeof(float));
    if (d_depth_in_) (void)hipMemset(d_depth_in_.get(), 0, depth_scratch_.size() * sizeof(uint16_t));
    if (d_depth_out_) (void)hipMemset(d_depth_out_.get(), 0, depth_scratch_.size() * sizeof(uint16_t));
#endif
}

void SignalConditioner::resetEMA() {
    std::fill(ema_buf_m_.begin(), ema_buf_m_.end(), 0.0f);
#ifdef CUDA_ENABLED
    if (d_ema_buf_m_) cudaMemset(d_ema_buf_m_.get(), 0, ema_buf_m_.size() * sizeof(float));
#elif defined(HIP_ENABLED)
    if (d_ema_buf_m_) (void)hipMemset(d_ema_buf_m_.get(), 0, ema_buf_m_.size() * sizeof(float));
#endif
}

void SignalConditioner::process(RawFrame& raw, cudaStream_t cuda_stream, float min_depth_m, float max_depth_m) {
    (void)cuda_stream;

    if (raw.rgb.size() != sr_rgb_.size() || raw.depth.size() != depth_scratch_.size()) {
        return;
    }

#ifdef CUDA_ENABLED
    if (cuda_stream && processCuda(raw, cuda_stream, min_depth_m, max_depth_m)) {
        return;
    }
#elif defined(HIP_ENABLED)
    if (cuda_stream && processCuda(raw, cuda_stream, min_depth_m, max_depth_m)) {
        return;
    }
#endif

    processCpu(raw, min_depth_m, max_depth_m);
}

void SignalConditioner::processCpu(RawFrame& raw, float min_depth_m, float max_depth_m) {
    static int frame_counter = 0;
    static bool logging_initialized = false;
    
    if (!logging_initialized) {
        initLogging();
        logging_initialized = true;
    }
    
    preprocessRgb(raw.rgb);
    applySuperResolutionToRgb(raw.rgb); // Apply EASU+RCAS for TSDF texturing
    buildSuperResolutionGuidance(raw.rgb); // Build guidance from processed RGB
    denoiseDepthSpatial(raw.depth, min_depth_m, max_depth_m);
    fillDepthHoles(raw.depth, min_depth_m, max_depth_m);
    guidedDepthFilter(raw.depth, min_depth_m, max_depth_m);
    applyDepthEma(raw.depth, min_depth_m, max_depth_m);
    
    if (g_logging_enabled) {
        logFrameStats(frame_counter++);
    }
}

void SignalConditioner::preprocessRgb(std::vector<uint8_t>& rgb) {
    bilateralDenoiseRgb(rgb, rgb_scratch_);
    rgb.swap(rgb_scratch_);

    // Block-based CLAHE removed because independent 8x8 tiled histogram equalization
    // without bilinear interpolation causes false grid boundary artifacts that corrupt
    // the downstream guided depth filter. We rely on the `medianBlur3x3` for noise 
    // reduction and `applyCAS` (SuperResolution) later for contrast enhancement.

    medianBlur3x3(rgb);
}

void SignalConditioner::buildSuperResolutionGuidance(const std::vector<uint8_t>& rgb) {
    // Keep guidance at original resolution for depth filtering
    // Apply RCAS sharpening for guidance quality
    sr_rgb_ = rgb;
    sr::applyCAS(sr_rgb_, FRAME_W, FRAME_H, 0.5f);

    // Build guidance luma from the sharpened image (original resolution)
    guidance_luma_.resize(FRAME_W * FRAME_H);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < FRAME_W * FRAME_H; ++i) {
        guidance_luma_[i] = rgbLuma(&sr_rgb_[i * 3]) / 255.0f;
    }
}

void SignalConditioner::applySuperResolutionToRgb(const std::vector<uint8_t>& rgb) {
    // Apply EASU upscaling + RCAS sharpening for TSDF texturing
    // This is separate from guidance to avoid resolution mismatch
    if (sr_scale_ > 1) {
#ifdef CUDA_ENABLED
        static const bool sr_warned = [] {
            KFLOG_WARN("SR", "EASU upscaling falls back to CPU on the CUDA backend (GPU EASU exists only on HIP)");
            return true;
        }();
        (void)sr_warned;
#endif
        // EASU upscaling
        std::vector<uint8_t> upscaled;
        sr::applyEASU(rgb, upscaled, FRAME_W, FRAME_H, sr_scale_);
        
        // RCAS sharpening on upscaled image
        int sr_w = FRAME_W * sr_scale_;
        int sr_h = FRAME_H * sr_scale_;
        sr::applyCAS(upscaled, sr_w, sr_h, 0.5f);
        
        sr_rgb_upscaled_ = upscaled;
    } else {
        // Just RCAS sharpening at original resolution
        sr_rgb_upscaled_ = rgb;
        sr::applyCAS(sr_rgb_upscaled_, FRAME_W, FRAME_H, 0.5f);
    }
}

void SignalConditioner::denoiseDepthSpatial(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
    depth_scratch_ = depth;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int idx = y * FRAME_W + x;
            const float center_depth_m = rawDepthToMeters(depth[idx]);
            if (!isValidDepthMeters(center_depth_m, min_depth_m, max_depth_m)) {
                continue;
            }

            float window[9];
            int count = 0;
            for (int dy = -kDepthMedianRadius; dy <= kDepthMedianRadius; ++dy) {
                const int sy = reflectCoord(y + dy, FRAME_H);
                for (int dx = -kDepthMedianRadius; dx <= kDepthMedianRadius; ++dx) {
                    const int sx = reflectCoord(x + dx, FRAME_W);
                    const float sample_depth_m = rawDepthToMeters(depth[sy * FRAME_W + sx]);
                    if (!isValidDepthMeters(sample_depth_m, min_depth_m, max_depth_m)) {
                        continue;
                    }
                    window[count++] = sample_depth_m;
                }
            }

            if (count >= 3) {
                std::nth_element(window, window + count / 2, window + count);
                depth_scratch_[idx] = metersToRawDepth(window[count / 2]);
            }
        }
    }

    depth.swap(depth_scratch_);
    if (g_logging_enabled) {
        #pragma omp atomic
        g_stats.median_filtered += FRAME_W * FRAME_H;
    }
}

void SignalConditioner::applyDepthEma(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < FRAME_W * FRAME_H; ++i) {
        const float depth_m = rawDepthToMeters(depth[i]);
        if (!isValidDepthMeters(depth_m, min_depth_m, max_depth_m) || std::isnan(depth_m) || std::isinf(depth_m)) {
            ema_buf_m_[i] = 0.0f; // Prevent ghosting by resetting stale EMA values
            if (g_logging_enabled) {
                #pragma omp atomic
                g_stats.ema_reset++;
            }
            continue;
        }

        const float previous = ema_buf_m_[i];
        const float delta = std::abs(depth_m - previous);
        
        // Improved EMA stability: check for temporal consistency with neighbors
        bool reset_ema = (std::isnan(previous) || std::isinf(previous) || previous <= 0.0f || delta > kEmaJumpResetMeters);
        
        // Additional validation: if delta is suspiciously large, check neighboring pixels
        if (!reset_ema && delta > kEmaJumpResetMeters * 0.5f) {
            const int x = i % FRAME_W;
            const int y = i / FRAME_W;
            float neighbor_sum = 0.0f;
            int neighbor_count = 0;
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int nx = reflectCoord(x + dx, FRAME_W);
                    const int ny = reflectCoord(y + dy, FRAME_H);
                    const int ni = ny * FRAME_W + nx;
                    const float nd = rawDepthToMeters(depth[ni]);
                    if (isValidDepthMeters(nd, min_depth_m, max_depth_m)) {
                        neighbor_sum += nd;
                        neighbor_count++;
                    }
                }
            }
            
            if (neighbor_count > 0) {
                const float neighbor_avg = neighbor_sum / neighbor_count;
                const float neighbor_delta = std::abs(depth_m - neighbor_avg);
                // If current depth is very different from neighbors, reset EMA
                if (neighbor_delta > kEmaJumpResetMeters) {
                    reset_ema = true;
                }
            }
        }
        
        const float filtered = reset_ema ? depth_m : (0.7f * depth_m + 0.3f * previous);

        ema_buf_m_[i] = filtered;
        depth[i] = metersToRawDepth(filtered);
        
        if (g_logging_enabled) {
            #pragma omp atomic
            g_stats.ema_filtered++;
            if (reset_ema) {
                #pragma omp atomic
                g_stats.ema_reset++;
            }
            #pragma omp critical
            {
                g_stats.max_depth_delta = std::max(g_stats.max_depth_delta, delta);
                g_stats.avg_depth_delta += delta;
            }
        }
    }
    if (g_logging_enabled && FRAME_W * FRAME_H > 0) {
        g_stats.avg_depth_delta /= (FRAME_W * FRAME_H);
    }
}

void SignalConditioner::fillDepthHoles(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
    // NOTE: depth_scratch_ is a member variable used as a write-scratch buffer.
    // This function must only be called from a single thread per SignalConditioner
    // instance. PipelineController guarantees this via the single trackingLoop thread.
    depth_scratch_ = depth;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int idx = y * FRAME_W + x;
            if (depth[idx] != 0) {
                continue;
            }

            // Edge-aware hole filling: use weighted average instead of nearest neighbor
            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;
            int valid_neighbors = 0;

            for (int dy = -kHoleFillRadius; dy <= kHoleFillRadius; ++dy) {
                const int sy = reflectCoord(y + dy, FRAME_H);
                for (int dx = -kHoleFillRadius; dx <= kHoleFillRadius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int sx = reflectCoord(x + dx, FRAME_W);

                    const float candidate_m = rawDepthToMeters(depth[sy * FRAME_W + sx]);
                    if (!isValidDepthMeters(candidate_m, min_depth_m, max_depth_m)) {
                        continue;
                    }

                    const int dist_sq = dx * dx + dy * dy;
                    const float weight = 1.0f / (dist_sq + 1.0f); // Inverse distance weighting
                    weighted_sum += weight * candidate_m;
                    weight_sum += weight;
                    valid_neighbors++;
                }
            }

            if (valid_neighbors > 0 && weight_sum > 1e-6f) {
                depth_scratch_[idx] = metersToRawDepth(weighted_sum / weight_sum);
                if (g_logging_enabled) {
                    #pragma omp atomic
                    g_stats.hole_filled++;
                }
            }
        }
    }

    depth.swap(depth_scratch_);
}

void SignalConditioner::guidedDepthFilter(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
    // PERFORMANCE NOTE: kGuidedRadius=9 produces a 19x19 kernel (~361 samples/pixel).
    // At 640x480 this is ~110M multiply-adds per frame on the CPU path. If real-time
    // performance is required, consider reducing kGuidedRadius to 4-5, or replacing
    // with a separable bilateral approximation.
    depth_scratch_ = depth;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int idx = y * FRAME_W + x;
            const float center_depth = rawDepthToMeters(depth[idx]);
            
            if (center_depth <= 0.01f) {
                depth_scratch_[idx] = 0;
                continue;
            }
            
            // Prevent guided filter from treating empty space as a massive hole-filler 
            // which smears sharp object geometry boundaries into the void.
            if (!isValidDepthMeters(center_depth, min_depth_m, max_depth_m)) {
                continue;
            }

            const float center_luma = guidance_luma_[idx];
            
            // Edge-aware filtering: compute depth gradient to reduce filter strength near edges
            const float edge_strength = computeDepthGradient(depth, x, y, FRAME_W, FRAME_H);
            if (g_logging_enabled && edge_strength > 0.5f) {
                #pragma omp atomic
                g_stats.edge_pixels++;
            }

            float sum_weights = 0.0f;
            float sum_depth = 0.0f;

            for (int dy = -kGuidedRadius; dy <= kGuidedRadius; ++dy) {
                const int sy = reflectCoord(y + dy, FRAME_H);
                for (int dx = -kGuidedRadius; dx <= kGuidedRadius; ++dx) {
                    const int sx = reflectCoord(x + dx, FRAME_W);

                    const int nidx = sy * FRAME_W + sx;
                    const float neighbor_depth = rawDepthToMeters(depth[nidx]);
                    if (!isValidDepthMeters(neighbor_depth, min_depth_m, max_depth_m)) {
                        continue;
                    }

                    const float spatial_dist_sq = static_cast<float>(dx * dx + dy * dy);
                    const float guidance_delta = guidance_luma_[nidx] - center_luma;
                    const float depth_delta = (center_depth > 0.0f) ? (neighbor_depth - center_depth) : 0.0f;

                    const float spatial_weight = std::exp(-spatial_dist_sq / (2.0f * kGuidedRadius * kGuidedRadius));
                    const float luma_weight = std::exp(-(guidance_delta * guidance_delta) / (2.0f * kGuidedSigmaLuma * kGuidedSigmaLuma + 0.01f));
                    const float depth_weight = std::exp(-(depth_delta * depth_delta) / (2.0f * kGuidedSigmaDepth * kGuidedSigmaDepth + 0.01f));
                    
                    // Edge-aware: reduce filter strength near edges to prevent blurring
                    const float edge_factor = 1.0f - edge_strength * 0.5f; // Reduce by up to 50% near edges
                    const float weight = spatial_weight * luma_weight * depth_weight * edge_factor;

                    sum_weights += weight;
                    sum_depth += weight * neighbor_depth;
                }
            }

            if (sum_weights > 1e-6f) {
                depth_scratch_[idx] = metersToRawDepth(sum_depth / sum_weights);
                if (g_logging_enabled) {
                    #pragma omp atomic
                    g_stats.guided_filtered++;
                }
            } else {
                // Fallback to original depth if filtering failed
                depth_scratch_[idx] = depth[idx];
            }
        }
    }

    depth.swap(depth_scratch_);
}

} // namespace sensor
} // namespace kfusion
