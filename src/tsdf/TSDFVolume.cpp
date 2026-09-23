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
    // Voxel-projective integration (KinectFusion Alg. 1; the CUDA kernel's
    // scheme). Every voxel inside the camera frustum's bounding box projects
    // its CORNER position origin + i*vs (the convention getTSDF, marching cubes
    // and the raycast all read back) into the depth image and takes exactly one
    // update per frame. One writer per voxel makes the result independent of
    // thread count and schedule with no sort or serial fold; the old per-pixel
    // ray march wrote each voxel ~20-40 times per frame (so max_weight counted
    // ray samples, not frames) and stored an SDF sampled at a different point
    // than the voxel it landed in, a half-voxel bias that drove tracking drift.
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

    // Frustum AABB in voxel coordinates: camera centre plus the four image
    // corners pushed to the far limit (max_depth + trunc). Voxels outside it
    // cannot project into the image with a depth in range.
    Eigen::Vector3f lo = (t_cw - origin) / vs;
    Eigen::Vector3f hi = lo;
    const float z_far = max_depth + trunc;
    const float corner_px[4][2] = {{0.0f, 0.0f}, {float(width), 0.0f},
                                   {0.0f, float(height)}, {float(width), float(height)}};
    for (const auto& c : corner_px) {
        const Eigen::Vector3f ray((c[0] - cx) / fx * z_far, (c[1] - cy) / fy * z_far, z_far);
        const Eigen::Vector3f v = (R_cw * ray + t_cw - origin) / vs;
        lo = lo.cwiseMin(v);
        hi = hi.cwiseMax(v);
    }
    if (!lo.allFinite() || !hi.allFinite()) return;
    const auto clampIdx = [res](float f) {
        return static_cast<int>(std::max(0.0f, std::min(static_cast<float>(res - 1), f)));
    };
    const int x0 = clampIdx(std::floor(lo.x())), x1 = clampIdx(std::ceil(hi.x()));
    const int y0 = clampIdx(std::floor(lo.y())), y1 = clampIdx(std::ceil(hi.y()));
    const int z0 = clampIdx(std::floor(lo.z())), z1 = clampIdx(std::ceil(hi.z()));
    const int ny = y1 - y0 + 1;
    const int rows = (z1 - z0 + 1) * ny;

    // Per-row partial diagnostics, reduced below in row order so an
    // AZU_TSDF_LOG=1 run is as deterministic as the volume itself.
    const bool logging = csvSink().enabled;
    struct RowStats {
        int    updated = 0, colored = 0, clamped = 0;
        float  max_abs = 0.0f;
        double sum_abs = 0.0;
    };
    std::vector<RowStats> row_stats(logging ? static_cast<size_t>(rows) : 0u);

    const Eigen::Vector3f step_x = R_wc.col(0) * vs;
    const float color_band = 0.5f * trunc;

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int z = z0; z <= z1; ++z) {
        for (int y = y0; y <= y1; ++y) {
            RowStats rs;
            Eigen::Vector3f c = R_wc * (origin + Eigen::Vector3f(float(x0), float(y), float(z)) * vs) + t_wc;
            for (int x = x0; x <= x1; ++x, c += step_x) {
                if (c.z() <= 0.0f) continue;
                const float inv_z = 1.0f / c.z();
                const float uf = std::floor(fx * c.x() * inv_z + cx + 0.5f);
                const float vf = std::floor(fy * c.y() * inv_z + cy + 0.5f);
                if (!(uf >= 0.0f && uf < static_cast<float>(width) &&
                      vf >= 0.0f && vf < static_cast<float>(height))) continue;
                const int pix = static_cast<int>(vf) * width + static_cast<int>(uf);

                const float d = depth_meters[pix];
                if (!(d >= min_depth && d <= max_depth)) continue; // also rejects NaN
                const float sdf = d - c.z();
                if (sdf < -trunc) continue;
                const float tsdf_new = std::min(1.0f, sdf / trunc);

                Voxel& vox = voxels_[idx(x, y, z)];
                const float w_old = vox.weight;
                vox.tsdf   = (vox.tsdf * w_old + tsdf_new) / (w_old + 1.0f);
                vox.weight = std::min(w_old + 1.0f, max_w);

                const bool colored = rgb != nullptr && std::fabs(sdf) < color_band;
                if (colored) {
                    // Canonical fold (docs/CANONICAL_SEMANTICS.md, "Color
                    // pipeline"): device byte -> float sRGB, blended with the
                    // same weights as tsdf. Bytes are finite, so the blend is.
                    const uint8_t* px = rgb + pix * 3;
                    const float inv = 1.0f / (w_old + 1.0f);
                    vox.r = (vox.r * w_old + utils::srgbUint8ToFloat(px[0])) * inv;
                    vox.g = (vox.g * w_old + utils::srgbUint8ToFloat(px[1])) * inv;
                    vox.b = (vox.b * w_old + utils::srgbUint8ToFloat(px[2])) * inv;
                }
                if (logging) {
                    ++rs.updated;
                    rs.colored += colored ? 1 : 0;
                    rs.clamped += (tsdf_new >= 1.0f) ? 1 : 0;
                    rs.max_abs = std::max(rs.max_abs, std::fabs(sdf));
                    rs.sum_abs += std::fabs(sdf);
                }
            }
            if (logging) row_stats[static_cast<size_t>((z - z0) * ny + (y - y0))] = rs;
        }
    }

    if (logging) {
        for (int i = 0; i < width * height; ++i) {
            const float d = depth_meters[i];
            if (!(d >= min_depth && d <= max_depth)) ++stats_.depth_filtered;
        }
        for (const RowStats& rs : row_stats) {
            stats_.voxels_updated     += rs.updated;
            stats_.color_updates      += rs.colored;
            stats_.truncation_clamped += rs.clamped;
            stats_.max_abs_sdf         = std::max(stats_.max_abs_sdf, rs.max_abs);
            stats_.sum_abs_sdf        += rs.sum_abs;
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
