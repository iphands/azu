#include "sensor/SuperResolution.h"

#include "sensor/BorderMode.h"  // cpuReflectCoord: the one CPU border-mode owner

#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace sensor {
namespace sr {

// ---------------------------------------------------------------------------
// FSR 1.0 EASU — Edge Adaptive Spatial Upsampling (CPU Reference)
//
// Paper: "FidelityFX Super Resolution" — AMD (2021)
// This is the EASU upscaling pass, NOT the RCAS sharpening pass.
//
// Algorithm: Bicubic interpolation with edge-adaptive sampling
// ---------------------------------------------------------------------------

inline void getPixelF(const std::vector<uint8_t>& img,
                      int x, int y, int w, int h,
                      float out[3]) {
    x = cpuReflectCoord(x, w);
    y = cpuReflectCoord(y, h);
    int idx = (y * w + x) * 3;
    out[0] = img[idx + 0] / 255.0f;
    out[1] = img[idx + 1] / 255.0f;
    out[2] = img[idx + 2] / 255.0f;
}

// Bicubic interpolation kernel (Catmull-Rom)
inline float cubic(float x) {
    float ax = std::abs(x);
    if (ax <= 1.0f) {
        return 0.5f * (ax * ax * ax - 2.0f * ax * ax + 1.0f);
    } else if (ax < 2.0f) {
        return 0.5f * (2.0f - ax) * (2.0f - ax) * (2.0f - ax);
    }
    return 0.0f;
}

void applyEASU_CPU(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                   int src_width, int src_height, int scale) {
    if (src.empty() || src_width <= 0 || src_height <= 0 || scale < 2 || scale > 4) return;

    int dst_width = src_width * scale;
    int dst_height = src_height * scale;
    dst.resize(dst_width * dst_height * 3);

    float x_scale = static_cast<float>(src_width) / dst_width;
    float y_scale = static_cast<float>(src_height) / dst_height;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dst_height; ++y) {
        for (int x = 0; x < dst_width; ++x) {
            // Map destination pixel to source coordinates
            float src_x = (x + 0.5f) * x_scale - 0.5f;
            float src_y = (y + 0.5f) * y_scale - 0.5f;

            int x0 = static_cast<int>(std::floor(src_x)) - 1;
            int y0 = static_cast<int>(std::floor(src_y)) - 1;

            float sum[3] = {0.0f, 0.0f, 0.0f};
            float weight_sum = 0.0f;

            // 4x4 bicubic sampling
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    int sample_x = x0 + i;
                    int sample_y = y0 + j;

                    float px[3];
                    getPixelF(src, sample_x, sample_y, src_width, src_height, px);

                    float wx = cubic(src_x - (x0 + i));
                    float wy = cubic(src_y - (y0 + j));
                    float w = wx * wy;

                    for (int ch = 0; ch < 3; ++ch) {
                        sum[ch] += px[ch] * w;
                    }
                    weight_sum += w;
                }
            }

            int dst_idx = (y * dst_width + x) * 3;
            for (int ch = 0; ch < 3; ++ch) {
                float val = (weight_sum > 1e-6f) ? (sum[ch] / weight_sum) : 0.0f;
                dst[dst_idx + ch] = static_cast<uint8_t>(std::clamp(val, 0.0f, 1.0f) * 255.0f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FSR 1.0 RCAS — Robust Contrast Adaptive Sharpening (CPU Reference)
//
// Paper: "FidelityFX Super Resolution" — AMD (2021)
// This is the RCAS sharpening pass, NOT the EASU upscaling pass.
//
// Algorithm (matches A_CPU reference in ffx_fsr1.h):
//   For each pixel centre c with cross-neighbors a(top), b(left), d(right), e(bottom):
//     For each channel ch in {R,G,B}:
//       mn_c  = min(a_c, b_c, c_c, d_c, e_c)
//       mx_c  = max(a_c, b_c, c_c, d_c, e_c)
//       amp_c = min(mn_c, 1-mx_c) / mx_c        [contrast-adaptive limit, per channel]
//     amp   = min(amp_R, amp_G, amp_B)           [most restrictive channel wins]
//     w     = amp * peak                         [peak = mapped sharpness param]
//     out_c = (c_c + w*(a_c+b_c+d_c+e_c)) / (1 + 4*w)
// ---------------------------------------------------------------------------

void applyCAS_CPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness) {
    if (rgb.empty() || width <= 0 || height <= 0) return;

    std::vector<uint8_t> scratch(rgb.size());

    // Map sharpness [0.0, 1.0] → FSR peak [-1/8, -1/5]
    const float t    = std::clamp(sharpness, 0.0f, 1.0f);
    const float peak = -1.0f / ((1.0f - t) * 8.0f + t * 5.0f);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float a[3], b[3], c[3], d[3], e[3];
            getPixelF(rgb, x,     y - 1, width, height, a); // Top
            getPixelF(rgb, x - 1, y,     width, height, b); // Left
            getPixelF(rgb, x,     y,     width, height, c); // Centre
            getPixelF(rgb, x + 1, y,     width, height, d); // Right
            getPixelF(rgb, x,     y + 1, width, height, e); // Bottom

            // Per-channel min/max over the cross-pattern
            float amp = 1.0f; // will be clamped to the most restrictive channel
            for (int ch = 0; ch < 3; ++ch) {
                float mn = std::min({a[ch], b[ch], c[ch], d[ch], e[ch]});
                float mx = std::max({a[ch], b[ch], c[ch], d[ch], e[ch]});
                // Contrast-adaptive amplitude: how much headroom exists?
                float mx_safe = std::max(mx, 1e-6f);
                float amp_ch = std::min(mn, 1.0f - mx) / mx_safe;
                amp = std::min(amp, amp_ch);
            }

            float w          = amp * peak;
            float weight_sum = 1.0f + 4.0f * w;

            for (int ch = 0; ch < 3; ++ch) {
                float val = (c[ch] + w * (a[ch] + b[ch] + d[ch] + e[ch])) / weight_sum;
                scratch[(y * width + x) * 3 + ch] =
                    static_cast<uint8_t>(std::clamp(val, 0.0f, 1.0f) * 255.0f);
            }
        }
    }

    std::memcpy(rgb.data(), scratch.data(), rgb.size());
}

} // namespace sr
} // namespace sensor
} // namespace kfusion
