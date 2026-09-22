#include "tsdf/TSDFVolume.h"
#include "utils/ColorMath.h"
#include "utils/CoordinateMath.h"
#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <fstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace tsdf {

namespace {

// Process-global CSV diagnostic channel for TSDF integration (AZU_TSDF_LOG=1).
// It is a deliberately process-wide facility, NOT per-instance state: one file,
// one header line, opened at most once. The C++11 function-local static makes
// the read-mostly configuration write-once and thread-safe (magic-static
// initialization runs exactly once even if two threads integrate at once), which
// is why there is no mutable `logging_enabled` / `initialized` flag pair racing
// on the hot path any more. The per-frame COUNTERS are not here — they live in
// TSDFVolume::Stats, the object that produces them.
struct CsvSink {
    std::ofstream file;
    bool          enabled = false;

    CsvSink() {
        const char* log_env = std::getenv("AZU_TSDF_LOG");
        if (log_env && std::string(log_env) == "1") {
            enabled = true;
            file.open("tsdf_integration_log.csv");
            if (file.is_open()) {
                file << "frame,depth_filtered,voxels_updated,color_updates,"
                        "truncation_clamped,max_sdf,avg_sdf\n";
                file.flush();
            } else {
                enabled = false;   // fail closed: no header, no rows
            }
        }
    }
};

CsvSink& csvSink() {
    static CsvSink sink;
    return sink;
}

} // namespace

void TSDFVolume::logDiagnostics() {
    CsvSink& sink = csvSink();
    if (!sink.enabled || !sink.file.is_open()) return;

    // avg_sdf divides by the CONTRIBUTING sample count, not by the raster size.
    // Every sample folded into the sum is one voxels_updated increment in the
    // same statement, so voxels_updated is exactly the denominator; a frame that
    // updated nothing has no mean at all and stays 0.0 instead of 0/0 = NaN.
    const float avg_sdf =
        (stats_.voxels_updated > 0)
            ? static_cast<float>(stats_.sum_abs_sdf / stats_.voxels_updated)
            : 0.0f;

    sink.file << stats_frame_++ << ","
              << stats_.depth_filtered << ","
              << stats_.voxels_updated << ","
              << stats_.color_updates << ","
              << stats_.truncation_clamped << ","
              << stats_.max_abs_sdf << ","
              << avg_sdf << "\n";
    sink.file.flush();
    stats_ = Stats{};
}

namespace {
// Exact comparison over every field, so a change in any one of them is visible.
// A NaN field compares unequal on purpose: the conservative outcome is to clear.
bool sameParams(const TSDFParams& a, const TSDFParams& b) {
    return a.resolution == b.resolution &&
           a.voxel_size == b.voxel_size &&
           a.truncation  == b.truncation &&
           a.max_weight  == b.max_weight &&
           a.min_depth == b.min_depth &&
           a.max_depth == b.max_depth &&
           (a.origin.array() == b.origin.array()).all();
}
} // namespace

TSDFVolume::TSDFVolume(const TSDFParams& params)
    : params_(params)
{
    size_t total = static_cast<size_t>(params.resolution) * params.resolution * params.resolution;
    voxels_.resize(total);
    reset();
}

TSDFVolume::~TSDFVolume() {
#ifdef CUDA_ENABLED
    freeGPU();
#elif defined(HIP_ENABLED)
    freeGPU();
#endif
}

void TSDFVolume::reset() {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    unlocked_reset();
}

void TSDFVolume::unlocked_reset() {
    std::fill(voxels_.begin(), voxels_.end(),
              Voxel{EMPTY_TSDF, EMPTY_WEIGHT, EMPTY_COLOR, EMPTY_COLOR, EMPTY_COLOR});
    integrated_frames_.store(0);
}

void TSDFVolume::setParams(const TSDFParams& p) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    if (sameParams(p, params_)) return;

    const bool resolution_changed = (p.resolution != params_.resolution);
    params_ = p;
    if (resolution_changed) {
        // KIN-FORK: extractGlobalPointCloud* allocates d_pc_* scratch lazily
        // (if(!ptr)) and never resizes -> stale small buffers become OOB once
        // the volume grows. Drop them so the next extract re-allocates at the
        // new size (same guard shape as ~TSDFVolume above).
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
        freeGPU();
#endif
        size_t total = static_cast<size_t>(params_.resolution) * params_.resolution * params_.resolution;
        voxels_.resize(total);
    }
    // Every parameter difference invalidates existing voxels, not only a
    // resolution change: tsdf/weight/color were fused under the old truncation
    // and max_weight, and world<->voxel mapping was derived from the old
    // voxel_size and origin. Do NOT call reset() here — the lock is held.
    unlocked_reset();
}

void TSDFVolume::integrate(const float*           depth_meters,
                           const uint8_t*         rgb,
                           const Eigen::Matrix4f& pose,
                           float fx, float fy,
                           float cx, float cy,
                           int   width, int height,
                           float min_depth,
                           float max_depth)
{
    std::unique_lock<std::shared_mutex> lk(mutex_);
#ifdef CUDA_ENABLED
    if (gpu_enabled_) {
        size_t depth_size = width * height;
        size_t rgb_size = width * height * 3;
        
        // Reallocate buffers if size changed (memory leak fix)
        if (!d_depth_integ_ || last_depth_size_ != depth_size) {
            d_depth_integ_ = utils::make_cuda_unique<float>(depth_size);
            last_depth_size_ = depth_size;
        }
        if (rgb && (!d_rgb_integ_ || last_rgb_size_ != rgb_size)) {
            d_rgb_integ_ = utils::make_cuda_unique<uint8_t>(rgb_size);
            last_rgb_size_ = rgb_size;
        }
        
        cudaMemcpy(d_depth_integ_.get(), depth_meters, width * height * sizeof(float), cudaMemcpyHostToDevice);
        if (rgb) {
            cudaMemcpy(d_rgb_integ_.get(), rgb, width * height * 3, cudaMemcpyHostToDevice);
        }
        integrateGPU(d_depth_integ_.get(), rgb ? d_rgb_integ_.get() : nullptr, pose, fx, fy, cx, cy, width, height, min_depth, max_depth);
    } else {
        integrateCPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height, min_depth, max_depth);
    }
#elif defined(HIP_ENABLED)
    if (gpu_enabled_) {
        size_t depth_size = width * height;
        size_t rgb_size = width * height * 3;
        
        // Reallocate buffers if size changed (memory leak fix)
        if (!d_depth_integ_ || last_depth_size_ != depth_size) {
            d_depth_integ_ = utils::make_hip_unique<float>(depth_size);
            last_depth_size_ = depth_size;
        }
        if (rgb && (!d_rgb_integ_ || last_rgb_size_ != rgb_size)) {
            d_rgb_integ_ = utils::make_hip_unique<uint8_t>(rgb_size);
            last_rgb_size_ = rgb_size;
        }
        
        (void)hipMemcpy(d_depth_integ_.get(), depth_meters, width * height * sizeof(float), hipMemcpyHostToDevice);
        if (rgb) {
            (void)hipMemcpy(d_rgb_integ_.get(), rgb, width * height * 3, hipMemcpyHostToDevice);
        }
        integrateGPU(d_depth_integ_.get(), rgb ? d_rgb_integ_.get() : nullptr, pose, fx, fy, cx, cy, width, height, min_depth, max_depth);
    } else {
        integrateCPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height, min_depth, max_depth);
    }
#else
    integrateCPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height, min_depth, max_depth);
#endif
    integrated_frames_.fetch_add(1);
    
    // Still under the unique_lock taken above: the diagnostics read + reset is
    // serialized with every other integration exactly like the writes are.
    logDiagnostics();
}

void TSDFVolume::integrateCPU(const float*           depth_meters,
                               const uint8_t*         rgb,
                               const Eigen::Matrix4f& pose,
                               float fx, float fy,
                               float cx, float cy,
                               int   width, int height,
                               float min_depth,
                               float max_depth)
{
    const float trunc = params_.truncation;
    const float max_w = params_.max_weight;
    const float vs    = params_.voxel_size;
    const int   res   = params_.resolution;
    const Eigen::Vector3f origin = params_.origin;

    const Eigen::Matrix4f world_to_cam = pose.inverse();
    const Eigen::Matrix3f R_wc = world_to_cam.block<3,3>(0,0);
    const Eigen::Vector3f t_wc = world_to_cam.block<3,1>(0,3);

    const Eigen::Matrix3f R_cw = pose.block<3,3>(0,0);
    const Eigen::Vector3f t_cw = pose.block<3,1>(0,3);

    // big-fix Todo 14: deterministic two-phase CPU integration. The old kernel ran
    // `#pragma omp parallel for` over pixels and did an unsynchronised read-modify-
    // write of voxels_[idx] from every thread, so overlapping ray marches raced on
    // weight/tsdf/color and the fused voxel changed run to run. Canonical fold order
    // is (increasing y, then x, then march step) — the order one thread visits the
    // (pixel, step) candidates. Phase 1 derives each candidate from its own (pixel,
    // step) alone (no shared write); Phase 2 applies them in that order through the
    // unchanged formulas, making the volume a pure function of the input.
    //
    // `key` is a lossless raster+march sequence number: high 32 bits = raster pixel
    // index (y*width+x, monotone in y then x), low 32 bits = march step, so an
    // ascending sort of `key` reproduces canonical (y, x, step) order no matter what
    // order threads appended their candidates.
    struct Candidate {
        uint64_t key;         // (pixel_index << 32) | step
        int      vidx;        // target voxel index, idx(vx, vy, vz)
        float    tsdf_new;    // min(1, sdf / trunc), already NaN/Inf-filtered
        float    abs_sdf;     // |sdf|, for deterministic logging accumulation only
        bool     apply_color; // rgb present && sdf > -trunc * 0.5f
    };

    // Logging never alters the canonical fold; when enabled its counters accumulate
    // deterministically (integer sums in Phase 1, the float |sdf| sum in Phase 2
    // canonical order), so an AZU_TSDF_LOG=1 run is reproducible too.
    const bool logging = csvSink().enabled;
    std::vector<Candidate> candidates;

    #pragma omp parallel
    {
        // Thread-local buffer: no lock, no per-pixel allocation; merged once per
        // thread. Bounded by in-band/in-bounds/non-rejected (pixel, step) pairs.
        std::vector<Candidate> local;
        int local_depth_filtered = 0;
        int local_truncation_clamped = 0;

        #pragma omp for schedule(dynamic, 16) nowait
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float d_meas = depth_meters[y * width + x];
                // Filter invalid depths: outside the configured band (FusionHyperparams
                // owns it; the old hard-coded 0.1f floor silently overrode it), or NaN/inf.
                if (d_meas < min_depth || d_meas > max_depth ||
                    std::isnan(d_meas) || std::isinf(d_meas)) {
                    if (logging) ++local_depth_filtered;
                    continue;
                }

                // Ray direction in camera space
                Eigen::Vector3f ray_cam((x - cx) / fx, (y - cy) / fy, 1.0f);
                float ray_dist_scale = ray_cam.norm(); // Euclidean / Z ratio
                ray_cam.normalize();

                // Convert Z-depth to Euclidean distance along the ray
                float t_meas = d_meas * ray_dist_scale;
                // t is a Euclidean ray parameter while the band is a Z-depth bound, so
                // the near clamp is scaled by the same ratio; the other end is the
                // truncation segment itself.
                float t_min  = std::max(min_depth * ray_dist_scale, t_meas - trunc);
                float t_max  = t_meas + trunc;

                // Convert to world space
                Eigen::Vector3f ray_w = R_cw * ray_cam;
                Eigen::Vector3f cam_pos_w = t_cw;

                const int pixel_index = y * width + x;
                int step = 0;
                // March through the truncation segment; `step` numbers each march
                // iteration ascending so it forms the low half of the sequence key.
                for (float t = t_min; t <= t_max; t += vs * 0.75f, ++step) {
                    Eigen::Vector3f wpos = cam_pos_w + ray_w * t;

                    // World to voxel index
                    Eigen::Vector3f vpos = (wpos - origin) / vs;
                    int vx = static_cast<int>(std::floor(vpos.x()));
                    int vy = static_cast<int>(std::floor(vpos.y()));
                    int vz = static_cast<int>(std::floor(vpos.z()));

                    if (vx < 0 || vx >= res || vy < 0 || vy >= res || vz < 0 || vz >= res)
                        continue;

                    // Project voxel back to camera plane to get precise Z-depth difference
                    Eigen::Vector3f cpos = R_wc * wpos + t_wc;
                    float sdf = d_meas - cpos.z();

                    if (sdf < -trunc) continue;

                    float tsdf_new = std::min(1.0f, sdf / trunc);
                    if (std::isnan(tsdf_new) || std::isinf(tsdf_new)) continue;

                    if (logging && (tsdf_new >= 1.0f || tsdf_new <= -1.0f))
                        ++local_truncation_clamped;

                    local.push_back(Candidate{
                        (static_cast<uint64_t>(pixel_index) << 32) |
                            static_cast<uint32_t>(step),
                        idx(vx, vy, vz),
                        tsdf_new,
                        std::abs(sdf),
                        rgb != nullptr && sdf > -trunc * 0.5f});
                }
            }
        }

        #pragma omp critical
        {
            candidates.insert(candidates.end(), local.begin(), local.end());
            if (logging) {
                stats_.depth_filtered     += local_depth_filtered;
                stats_.truncation_clamped += local_truncation_clamped;
            }
        }
    }

    // Phase 2: canonical serial fold. Candidates are applied in ascending raster +
    // march order through the exact weight/TSDF/color formulas the old kernel used,
    // so each voxel equals the single-threaded sequential update regardless of how
    // Phase 1 was scheduled. Only the pure per-pixel geometry (Phase 1) is parallel.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.key < b.key; });

    for (const Candidate& c : candidates) {
        Voxel& vox = voxels_[static_cast<size_t>(c.vidx)];

        float w_old = vox.weight;
        float w_new = 1.0f;
        float w_sum = std::min(w_old + w_new, max_w);

        float next_tsdf = (vox.tsdf * w_old + c.tsdf_new * w_new) / (w_old + w_new + 1e-6f);
        if (std::isnan(next_tsdf) || std::isinf(next_tsdf)) continue;

        vox.tsdf   = next_tsdf;
        vox.weight = w_sum;

        if (logging) {
            ++stats_.voxels_updated;
            stats_.max_abs_sdf = std::max(stats_.max_abs_sdf, c.abs_sdf);
            stats_.sum_abs_sdf += c.abs_sdf;
        }

        if (c.apply_color) {
            // Same (y, x) that produced this candidate → its own RGB sample.
            const int pidx = static_cast<int>(c.key >> 32) * 3;
            // Canonical fold (docs/CANONICAL_SEMANTICS.md, "Color pipeline"):
            // the incoming device byte becomes float sRGB and blends in the same
            // weighted-average denominator as tsdf/weight. No per-update rounding
            // or truncation here — that is what made a heavily-fused voxel stop
            // converging (a |delta| under half a byte was rounded away). The
            // byte stage is deliberately left to the extraction boundaries.
            const float denom = w_old + 1.0f + 1e-6f;
            const float r_next = (vox.r * w_old + utils::srgbUint8ToFloat(rgb[pidx + 0])) / denom;
            const float g_next = (vox.g * w_old + utils::srgbUint8ToFloat(rgb[pidx + 1])) / denom;
            const float b_next = (vox.b * w_old + utils::srgbUint8ToFloat(rgb[pidx + 2])) / denom;
            // Assign only when every channel is finite: a non-finite candidate
            // must leave the voxel's previous color intact rather than poison it,
            // exactly as the tsdf guard above skips a non-finite blend.
            if (std::isfinite(r_next) && std::isfinite(g_next) && std::isfinite(b_next)) {
                vox.r = r_next;
                vox.g = g_next;
                vox.b = b_next;
                if (logging) ++stats_.color_updates;
            }
        }
    }
}

void TSDFVolume::raycast(const Eigen::Matrix4f& pose,
                         float fx, float fy, float cx, float cy,
                         int width, int height,
                         Eigen::Vector3f* vertices_out,
                         Eigen::Vector3f* normals_out,
                         uint8_t* colors_out) const
{
    std::shared_lock<std::shared_mutex> lk(mutex_);

    const float vs    = params_.voxel_size;
    const Eigen::Vector3f cam_origin = pose.block<3,1>(0,3);
    const Eigen::Matrix3f R_cw       = pose.block<3,3>(0,0);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int out_idx = y * width + x;
            vertices_out[out_idx] = Eigen::Vector3f::Zero();
            normals_out[out_idx]  = Eigen::Vector3f::Zero();
            if (colors_out) {
                colors_out[out_idx*3+0] = 0;
                colors_out[out_idx*3+1] = 0;
                colors_out[out_idx*3+2] = 0;
            }

            Eigen::Vector3f ray_cam((x - cx) / fx, (y - cy) / fy, 1.0f);
            // The band is a camera-plane Z-depth bound; a point at ray parameter t sits
            // at camera depth t / |ray_cam|, so the band maps to t by that scale.
            const float t_near = params_.min_depth * ray_cam.norm();
            const float t_far  = params_.max_depth * ray_cam.norm();
            Eigen::Vector3f ray_world = (R_cw * ray_cam).normalized();

            // Uniform half-voxel march: the crossing is interpolated between two
            // adjacent samples, so a coarser step would miss features thinner than the
            // step, and the interpolation divisor must be the step actually taken.
            const float h = 0.5f * vs;

            float t_prev = t_near;
            float f_prev = getTSDF(cam_origin + ray_world * t_prev);
            if (!std::isfinite(f_prev)) continue;

            float t_hit = t_near;
            bool  found = (f_prev == 0.0f);
            for (float t = t_near + h; !found && t <= t_far; t += h) {
                const float f_cur = getTSDF(cam_origin + ray_world * t);
                if (!std::isfinite(f_cur)) { found = false; break; }
                if ((f_prev > 0.0f && f_cur <= 0.0f) || (f_prev < 0.0f && f_cur >= 0.0f)) {
                    t_hit = t_prev + (t - t_prev) * (f_prev / (f_prev - f_cur));
                    found = true;
                    break;
                }
                t_prev = t;
                f_prev = f_cur;
            }
            if (!found) continue;

            const Eigen::Vector3f hit_world = cam_origin + ray_world * t_hit;
            const Eigen::Vector3f normal    = computeNormal(hit_world);
            if (!std::isfinite(t_hit) || !hit_world.allFinite() || !normal.allFinite()) continue;

            vertices_out[out_idx] = hit_world;
            normals_out[out_idx]  = normal;
            if (colors_out) {
                // Color comes from the voxel the resolved hit actually falls in, and
                // only from a fused (weighted) one, so it cannot be sampled from a
                // neighbour the surface never reached.
                const Eigen::Vector3i vi = worldToVoxel(hit_world);
                if (inBounds(vi.x(), vi.y(), vi.z())) {
                    const Voxel& vox = voxels_[idx(vi.x(), vi.y(), vi.z())];
                    if (vox.weight > EMPTY_WEIGHT) {
                        // One shared quantization policy at the byte boundary: a
                        // finite color, however far outside [0,1], publishes its
                        // saturated byte. Only a non-finite voxel color is a bug
                        // upstream, and then the whole surface output for this pixel
                        // is withdrawn (the documented all-or-nothing raycast
                        // contract) instead of emitting a fabricated black.
                        uint8_t qc[3] = {0, 0, 0};
                        if (utils::srgbFloatToUint8(vox.r, qc[0]) &&
                            utils::srgbFloatToUint8(vox.g, qc[1]) &&
                            utils::srgbFloatToUint8(vox.b, qc[2])) {
                            colors_out[out_idx*3+0] = qc[0];
                            colors_out[out_idx*3+1] = qc[1];
                            colors_out[out_idx*3+2] = qc[2];
                        } else {
                            vertices_out[out_idx] = Eigen::Vector3f::Zero();
                            normals_out[out_idx]  = Eigen::Vector3f::Zero();
                        }
                    }
                }
            }
        }
    }
}

Eigen::Vector3f TSDFVolume::computeNormal(const Eigen::Vector3f& world_pos) const {
    const float vs = params_.voxel_size;
    Eigen::Vector3f n;
    n.x() = (getTSDF(world_pos + Eigen::Vector3f(vs, 0, 0)) - getTSDF(world_pos - Eigen::Vector3f(vs, 0, 0)));
    n.y() = (getTSDF(world_pos + Eigen::Vector3f(0, vs, 0)) - getTSDF(world_pos - Eigen::Vector3f(0, vs, 0)));
    n.z() = (getTSDF(world_pos + Eigen::Vector3f(0, 0, vs)) - getTSDF(world_pos - Eigen::Vector3f(0, 0, vs)));
    float len = n.norm();
    if (len > 1e-6f) return n / len;
    return Eigen::Vector3f::Zero();
}

float TSDFVolume::getTSDF(const Eigen::Vector3f& world_pos) const {
    const float vs = params_.voxel_size;
    const int res = params_.resolution;
    
    Eigen::Vector3f v = (world_pos - params_.origin) / vs;
    
    // Trilinear interpolation
    int x0 = static_cast<int>(std::floor(v.x()));
    int y0 = static_cast<int>(std::floor(v.y()));
    int z0 = static_cast<int>(std::floor(v.z()));
    
    if (x0 < 0 || x0 >= res - 1 || y0 < 0 || y0 >= res - 1 || z0 < 0 || z0 >= res - 1) {
        Eigen::Vector3i vi = worldToVoxel(world_pos);
        if (!inBounds(vi.x(), vi.y(), vi.z())) return EMPTY_TSDF;
        const Voxel& vox = voxels_[idx(vi.x(), vi.y(), vi.z())];
        return (vox.weight > 0.0f) ? vox.tsdf : EMPTY_TSDF;
    }

    float tx = v.x() - x0;
    float ty = v.y() - y0;
    float tz = v.z() - z0;

    auto getV = [&](int x, int y, int z) {
        const Voxel& vox = voxels_[idx(x, y, z)];
        return (vox.weight > 0.0f) ? vox.tsdf : EMPTY_TSDF;
    };

    float v000 = getV(x0, y0, z0);
    float v100 = getV(x0+1, y0, z0);
    float v010 = getV(x0, y0+1, z0);
    float v110 = getV(x0+1, y0+1, z0);
    float v001 = getV(x0, y0, z0+1);
    float v101 = getV(x0+1, y0, z0+1);
    float v011 = getV(x0, y0+1, z0+1);
    float v111 = getV(x0+1, y0+1, z0+1);

    float v00 = v000 * (1 - tx) + v100 * tx;
    float v01 = v001 * (1 - tx) + v101 * tx;
    float v10 = v010 * (1 - tx) + v110 * tx;
    float v11 = v011 * (1 - tx) + v111 * tx;

    float v0 = v00 * (1 - ty) + v10 * ty;
    float v1 = v01 * (1 - ty) + v11 * ty;

    return v0 * (1 - tz) + v1 * tz;
}

const Voxel& TSDFVolume::voxelAt(int x, int y, int z) const {
    assert(inBounds(x, y, z));
    return voxels_[idx(x, y, z)];
}

Voxel& TSDFVolume::voxelAt(int x, int y, int z) {
    assert(inBounds(x, y, z));
    return voxels_[idx(x, y, z)];
}

Eigen::Vector3i TSDFVolume::worldToVoxel(const Eigen::Vector3f& world) const {
    Eigen::Vector3f v = (world - params_.origin) / params_.voxel_size;
    // Floor each axis independently through the shared helper (CPU canonical
    // rounding, docs/CANONICAL_SEMANTICS.md), so negative coordinates round
    // toward -inf exactly like the trilinear sampler in getTSDF() and the march
    // in integrateCPU() already do. A non-finite or out-of-int-range axis yields
    // the out-of-bounds sentinel, which inBounds() (and every caller, including
    // the getTSDF fallback) rejects, instead of the old static_cast<int> that
    // truncated toward zero and aliased a below-origin point to voxel (0,0,0).
    int vx = 0, vy = 0, vz = 0;
    if (!utils::floorToInt(v.x(), &vx) ||
        !utils::floorToInt(v.y(), &vy) ||
        !utils::floorToInt(v.z(), &vz)) {
        return Eigen::Vector3i(INT_MIN, INT_MIN, INT_MIN);
    }
    return Eigen::Vector3i(vx, vy, vz);
}

Eigen::Vector3f TSDFVolume::voxelToWorld(const Eigen::Vector3i& v) const {
    return voxelToWorld(v.x(), v.y(), v.z());
}

Eigen::Vector3f TSDFVolume::voxelToWorld(int x, int y, int z) const {
    return params_.origin + Eigen::Vector3f(x, y, z) * params_.voxel_size;
}

float TSDFVolume::usageFraction() const {
    size_t count = 0;
    for (const auto& v : voxels_) {
        if (v.weight > 0.0f) count++;
    }
    return static_cast<float>(count) / voxels_.size();
}

void TSDFVolume::extractGlobalPointCloud(std::vector<Eigen::Vector3f>& points_out,
                                         std::vector<uint8_t>&         colors_out) const
{
#ifdef CUDA_ENABLED
    if (gpu_enabled_) {
        extractGlobalPointCloudGPU(points_out, colors_out);
        return;
    }
#elif defined(HIP_ENABLED)
    if (gpu_enabled_) {
        extractGlobalPointCloudGPU(points_out, colors_out);
        return;
    }
#endif
    std::shared_lock<std::shared_mutex> lk(mutex_);
    const int res = params_.resolution;
    
    points_out.clear();
    colors_out.clear();

    // Estimate capacity to avoid excessive reallocations (~5% of volume)
    points_out.reserve(voxels_.size() / 20);
    colors_out.reserve(voxels_.size() / 20 * 3);

    #pragma omp parallel
    {
        std::vector<Eigen::Vector3f> local_points;
        std::vector<uint8_t>         local_colors;

        #pragma omp for nowait
        for (int z = 0; z < res; ++z) {
            for (int y = 0; y < res; ++y) {
                for (int x = 0; x < res; ++x) {
                    const Voxel& vox = voxels_[idx(x, y, z)];
                    // Only extract voxels near the surface (low absolute TSDF) with significant weight
                    if (vox.weight > 1.0f && std::abs(vox.tsdf) < 0.2f) {
                        // Same shared quantization policy as the raycast: a finite
                        // color saturates to its byte, and only a voxel whose color
                        // is non-finite is skipped, so the point count and the color
                        // count stay in lockstep either way.
                        uint8_t qc[3] = {0, 0, 0};
                        if (!utils::srgbFloatToUint8(vox.r, qc[0]) ||
                            !utils::srgbFloatToUint8(vox.g, qc[1]) ||
                            !utils::srgbFloatToUint8(vox.b, qc[2])) {
                            continue;
                        }
                        local_points.push_back(voxelToWorld(x, y, z));
                        local_colors.push_back(qc[0]);
                        local_colors.push_back(qc[1]);
                        local_colors.push_back(qc[2]);
                    }
                }
            }
        }

        #pragma omp critical
        {
            points_out.insert(points_out.end(), local_points.begin(), local_points.end());
            colors_out.insert(colors_out.end(), local_colors.begin(), local_colors.end());
        }
    }
}

} // namespace tsdf
} // namespace kfusion
