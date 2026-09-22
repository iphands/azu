#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "tracking/ICPShared.h"
#include <Eigen/Geometry>
#include <cstring>
#include <cmath>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace sensor {

// FrameData logging infrastructure
struct FrameDataStats {
    int depth_filtered = 0;
    int vertices_computed = 0;
    int normals_computed = 0;
    int normals_skipped_depth = 0;
    int normals_skipped_jump = 0;
    float max_depth = 0.0f;
    float avg_depth = 0.0f;
};

FrameDataStats g_frame_stats;
bool g_frame_logging_enabled = false;
std::ofstream g_frame_log_file;

void initFrameDataLogging() {
    const char* log_env = std::getenv("AZU_FRAME_LOG");
    if (log_env && std::string(log_env) == "1") {
        g_frame_logging_enabled = true;
        g_frame_log_file.open("framedata_log.csv");
        if (g_frame_log_file.is_open()) {
            g_frame_log_file << "frame,depth_filtered,vertices_computed,normals_computed,normals_skipped_depth,normals_skipped_jump,max_depth,avg_depth\n";
        }
    }
}

void logFrameDataStats(int frame_id) {
    if (!g_frame_logging_enabled || !g_frame_log_file.is_open()) return;
    g_frame_log_file << frame_id << ","
               << g_frame_stats.depth_filtered << ","
               << g_frame_stats.vertices_computed << ","
               << g_frame_stats.normals_computed << ","
               << g_frame_stats.normals_skipped_depth << ","
               << g_frame_stats.normals_skipped_jump << ","
               << g_frame_stats.max_depth << ","
               << g_frame_stats.avg_depth << "\n";
    g_frame_log_file.flush();
    // Reset stats for next frame
    g_frame_stats = FrameDataStats{};
}

namespace {

// The depth-discontinuity threshold is the SHARED one from
// include/tracking/ICPShared.h (big-fix Todo 12): the level-0 normal kernel and
// the pyramid downsample below consume one constant pair
// (kDepthJumpBaseMeters / kDepthJumpRelFrac) and cannot drift apart. The
// numeric values equal the literals this file used before Todo 12; only the
// definition site moved.
inline float depthJumpThreshold(float depth_m) {
    return kfusion::tracking::depthJumpThreshold(depth_m);
}

void updateVerticesFromDepth(FrameData& frame) {
    const int W = frame.width;
    const int H = frame.height;
    const float fx_inv = 1.0f / static_cast<float>(FX);
    const float fy_inv = 1.0f / static_cast<float>(FY);

    #pragma omp parallel for schedule(static) shared(frame, W, H, fx_inv, fy_inv)
    for (int idx = 0; idx < W * H; ++idx) {
        float d = frame.depth_meters[idx];
        if (d > 0.0f) {
            int x = idx % W;
            int y = idx / W;
            float vx = (static_cast<float>(x) - static_cast<float>(CX)) * fx_inv * d;
            float vy = (static_cast<float>(y) - static_cast<float>(CY)) * fy_inv * d;
            frame.vertices[idx] = Eigen::Vector3f(vx, vy, d);
        } else {
            frame.vertices[idx] = Eigen::Vector3f::Zero();
        }
    }
}

} // namespace

void buildFrameData(const uint16_t* raw_depth,
                    const uint8_t*  raw_rgb,
                    FrameData&      out,
                    float           min_depth,
                    float           max_depth)
{
    static int frame_counter = 0;
    static bool logging_initialized = false;
    
    if (!logging_initialized) {
        initFrameDataLogging();
        logging_initialized = true;
    }
    
    const int W = out.width;
    const int H = out.height;
    // THE CPU depth boundary for everything downstream (vertices, normals,
    // pyramid, ICP, TSDF): cpuDepthMeters() applies the canonical raw predicate
    // AND the configured [min_depth, max_depth] band, and returns exactly 0.0f
    // when either fails. The reciprocal curve has a pole at raw ~= 1084.61, so
    // a predicate-only check would let raw 1085 (-836.3 m) reach geometry.
    // Out-of-band depth is dropped to the invalid sentinel, never clamped to a
    // wall (big-fix Todo 20, docs/CANONICAL_SEMANTICS.md "Depth domain").
    #pragma omp parallel for schedule(static) shared(raw_depth, raw_rgb, out, min_depth, max_depth)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            float d = cpuDepthMeters(raw_depth[idx], min_depth, max_depth);
            if (d == 0.0f) {
                out.depth_meters[idx] = 0.0f;
                out.vertices[idx]     = Eigen::Vector3f::Zero();
                out.normals[idx]      = Eigen::Vector3f::Zero();
                if (g_frame_logging_enabled) {
                    #pragma omp atomic
                    g_frame_stats.depth_filtered++;
                }
            } else {
                out.depth_meters[idx] = d;
                out.vertices[idx] = Eigen::Vector3f::Zero();
                out.normals[idx]  = Eigen::Vector3f::Zero(); // computed separately
                
                if (g_frame_logging_enabled) {
                    #pragma omp atomic
                    g_frame_stats.vertices_computed++;
                    #pragma omp critical
                    {
                        g_frame_stats.max_depth = std::max(g_frame_stats.max_depth, d);
                        g_frame_stats.avg_depth += d;
                    }
                }
            }
            // Copy RGB
            out.rgb[idx * 3 + 0] = raw_rgb[idx * 3 + 0];
            out.rgb[idx * 3 + 1] = raw_rgb[idx * 3 + 1];
            out.rgb[idx * 3 + 2] = raw_rgb[idx * 3 + 2];
        }
    }

    updateVerticesFromDepth(out);
    
    if (g_frame_logging_enabled && W * H > 0) {
        g_frame_stats.avg_depth /= (W * H);
    }
    
    if (g_frame_logging_enabled) {
        logFrameDataStats(frame_counter++);
    }
}

void computeNormals(FrameData& frame) {
    const int W = frame.width;
    const int H = frame.height;

    #pragma omp parallel for schedule(static) shared(frame, W, H)
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int c  = y * W + x;
            int r  = y * W + (x + 1);
            int l  = y * W + (x - 1);
            int u  = (y - 1) * W + x;
            int d  = (y + 1) * W + x;

            if (frame.depth_meters[c] <= 0.0f ||
                frame.depth_meters[r] <= 0.0f ||
                frame.depth_meters[l] <= 0.0f ||
                frame.depth_meters[u] <= 0.0f ||
                frame.depth_meters[d] <= 0.0f) {
                frame.normals[c] = Eigen::Vector3f::Zero();
                if (g_frame_logging_enabled) {
                    #pragma omp atomic
                    g_frame_stats.normals_skipped_depth++;
                }
                continue;
            }

            const float dc = frame.depth_meters[c];
            const float jump = depthJumpThreshold(dc);
            if (std::abs(dc - frame.depth_meters[r]) > jump ||
                std::abs(dc - frame.depth_meters[l]) > jump ||
                std::abs(dc - frame.depth_meters[u]) > jump ||
                std::abs(dc - frame.depth_meters[d]) > jump) {
                frame.normals[c] = Eigen::Vector3f::Zero();
                if (g_frame_logging_enabled) {
                    #pragma omp atomic
                    g_frame_stats.normals_skipped_jump++;
                }
                continue;
            }

            Eigen::Vector3f dx = frame.vertices[r] - frame.vertices[l];
            Eigen::Vector3f dy = frame.vertices[d] - frame.vertices[u];
            Eigen::Vector3f n  = dx.cross(dy);
            float len = n.norm();
            if (len > 1e-6f) {
                frame.normals[c] = n / len;
                if (g_frame_logging_enabled) {
                    #pragma omp atomic
                    g_frame_stats.normals_computed++;
                }
            } else {
                frame.normals[c] = Eigen::Vector3f::Zero();
            }
        }
    }
}

// 2x box filter for downsampling depth + vertex + normal, with a depth-jump
// guard: a 2x2 block whose valid depths straddle a discontinuity larger than
// the shared max(0.03 m, 5% * d_min) threshold is NOT averaged (that blends
// both surfaces into a ghost vertex at every coarse pyramid level, the
// coarse-to-fine ICP's worst input). The nearest (smallest-depth) valid sample
// is kept whole instead — deterministic by first-minimum scan order. Blocks
// within the threshold average exactly as before, invalid/sentinel (0) samples
// are still excluded, and fully invalid blocks stay zero (big-fix Todo 12;
// docs/CANONICAL_SEMANTICS.md, dossier tracking:A34).
static FrameData downsample(const FrameData& src) {
    FrameData dst;
    dst.width  = src.width  / 2;
    dst.height = src.height / 2;
    int n = dst.width * dst.height;
    dst.vertices.assign(n, Eigen::Vector3f::Zero());
    dst.normals.assign(n, Eigen::Vector3f::Zero());
    dst.depth_meters.assign(n, 0.0f);
    dst.rgb.assign(n * 3, 0);

    const int SW = src.width;

    #pragma omp parallel for schedule(static) shared(src, dst, SW)
    for (int y = 0; y < dst.height; ++y) {
        for (int x = 0; x < dst.width; ++x) {
            int sx = x * 2;
            int sy = y * 2;

            int idx_dst = y * dst.width + x;

            // Collect valid samples from 2x2 block
            Eigen::Vector3f sum_v = Eigen::Vector3f::Zero();
            Eigen::Vector3f sum_n = Eigen::Vector3f::Zero();
            float sum_d = 0.0f;
            int count = 0;
            float d_min = 0.0f, d_max = 0.0f;  // valid-depth span
            int min_dx = 0, min_dy = 0;         // first-minimum sample position

            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int idx_src = (sy + dy) * SW + (sx + dx);
                    float d = src.depth_meters[idx_src];
                    if (d > 0.0f) {
                        sum_v += src.vertices[idx_src];
                        sum_n += src.normals[idx_src];
                        sum_d += d;
                        if (count == 0 || d < d_min) { d_min = d; min_dx = dx; min_dy = dy; }
                        if (count == 0 || d > d_max) { d_max = d; }
                        ++count;
                    }
                }
            }

            if (count > 0) {
                if (d_max - d_min > depthJumpThreshold(d_min)) {
                    // Discontinuity inside the block: keep the whole nearest
                    // sample rather than blending surfaces.
                    int idx_min = (sy + min_dy) * SW + (sx + min_dx);
                    dst.vertices[idx_dst]     = src.vertices[idx_min];
                    dst.depth_meters[idx_dst] = src.depth_meters[idx_min];
                    float nlen = src.normals[idx_min].norm();
                    dst.normals[idx_dst] = (nlen > 1e-6f)
                        ? Eigen::Vector3f(src.normals[idx_min] / nlen)
                        : Eigen::Vector3f::Zero();
                } else {
                    float inv = 1.0f / static_cast<float>(count);
                    dst.vertices[idx_dst]     = sum_v * inv;
                    dst.depth_meters[idx_dst] = sum_d * inv;
                    float nlen = sum_n.norm();
                    dst.normals[idx_dst] = (nlen > 1e-6f) ? Eigen::Vector3f(sum_n / nlen) : Eigen::Vector3f::Zero();
                }
            }

            // Nearest neighbor for RGB
            int src_idx = sy * SW + sx;
            dst.rgb[idx_dst * 3 + 0] = src.rgb[src_idx * 3 + 0];
            dst.rgb[idx_dst * 3 + 1] = src.rgb[src_idx * 3 + 1];
            dst.rgb[idx_dst * 3 + 2] = src.rgb[src_idx * 3 + 2];
        }
    }

    return dst;
}

void buildFramePyramid(const FrameData& full_res, FramePyramid& pyramid) {
    // Level 0: copy full res
    pyramid.levels[0] = full_res;

    // Level 1: half res
    pyramid.levels[1] = downsample(pyramid.levels[0]);

    // Level 2: quarter res
    pyramid.levels[2] = downsample(pyramid.levels[1]);
}

} // namespace sensor
} // namespace kfusion
