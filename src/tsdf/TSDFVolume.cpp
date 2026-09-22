#include "tsdf/TSDFVolume.h"
#include "utils/CoordinateMath.h"
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

// TSDF logging infrastructure
struct TSDFStats {
    int depth_filtered = 0;
    int voxels_updated = 0;
    int color_updates = 0;
    int truncation_clamped = 0;
    float max_sdf = 0.0f;
    float avg_sdf = 0.0f;
};

TSDFStats g_tsdf_stats;
bool g_tsdf_logging_enabled = false;
std::ofstream g_tsdf_log_file;

namespace {
// Exact comparison over every field, so a change in any one of them is visible.
// A NaN field compares unequal on purpose: the conservative outcome is to clear.
bool sameParams(const TSDFParams& a, const TSDFParams& b) {
    return a.resolution == b.resolution &&
           a.voxel_size == b.voxel_size &&
           a.truncation  == b.truncation &&
           a.max_weight  == b.max_weight &&
           (a.origin.array() == b.origin.array()).all();
}
} // namespace

void initTSDFLogging() {
    const char* log_env = std::getenv("AZU_TSDF_LOG");
    if (log_env && std::string(log_env) == "1") {
        g_tsdf_logging_enabled = true;
        g_tsdf_log_file.open("tsdf_integration_log.csv");
        if (g_tsdf_log_file.is_open()) {
            g_tsdf_log_file << "frame,depth_filtered,voxels_updated,color_updates,truncation_clamped,max_sdf,avg_sdf\n";
        }
    }
}

void logTSDFStats(int frame_id) {
    if (!g_tsdf_logging_enabled || !g_tsdf_log_file.is_open()) return;
    g_tsdf_log_file << frame_id << ","
               << g_tsdf_stats.depth_filtered << ","
               << g_tsdf_stats.voxels_updated << ","
               << g_tsdf_stats.color_updates << ","
               << g_tsdf_stats.truncation_clamped << ","
               << g_tsdf_stats.max_sdf << ","
               << g_tsdf_stats.avg_sdf << "\n";
    g_tsdf_log_file.flush();
    // Reset stats for next frame
    g_tsdf_stats = TSDFStats{};
}

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
    static int frame_counter = 0;
    static bool logging_initialized = false;
    
    if (!logging_initialized) {
        initTSDFLogging();
        logging_initialized = true;
    }
    
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
    
    if (g_tsdf_logging_enabled) {
        logTSDFStats(frame_counter++);
    }
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
    const bool logging = g_tsdf_logging_enabled;
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
                // Filter invalid depths: too close, too far, or NaN/inf.
                // Use 0.1f as minimum threshold (original behavior) to avoid
                // regression; the configured min_depth is owned by Todo 15, not here.
                if (d_meas < 0.1f || d_meas > max_depth ||
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
                float t_min  = std::max(0.1f, t_meas - trunc);
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
                g_tsdf_stats.depth_filtered     += local_depth_filtered;
                g_tsdf_stats.truncation_clamped += local_truncation_clamped;
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
            ++g_tsdf_stats.voxels_updated;
            g_tsdf_stats.max_sdf = std::max(g_tsdf_stats.max_sdf, c.abs_sdf);
            g_tsdf_stats.avg_sdf += c.abs_sdf;
        }

        if (c.apply_color) {
            // Same (y, x) that produced this candidate → its own RGB sample.
            const int pidx = static_cast<int>(c.key >> 32) * 3;
            // Use float arithmetic to avoid truncation darkening.
            vox.r = static_cast<uint8_t>(std::round((static_cast<float>(vox.r) * w_old + static_cast<float>(rgb[pidx+0])) / (w_old + 1.0f + 1e-6f)));
            vox.g = static_cast<uint8_t>(std::round((static_cast<float>(vox.g) * w_old + static_cast<float>(rgb[pidx+1])) / (w_old + 1.0f + 1e-6f)));
            vox.b = static_cast<uint8_t>(std::round((static_cast<float>(vox.b) * w_old + static_cast<float>(rgb[pidx+2])) / (w_old + 1.0f + 1e-6f)));

            if (logging) ++g_tsdf_stats.color_updates;
        }
    }
    
    if (g_tsdf_logging_enabled && width * height > 0) {
        g_tsdf_stats.avg_sdf /= (width * height);
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
            Eigen::Vector3f ray_world = (R_cw * ray_cam).normalized();

            float t = 0.3f; // min depth for Kinect v1
            float prev_tsdf = EMPTY_TSDF;
            const float trunc = params_.truncation;

            while (t < 5.0f) {
                Eigen::Vector3f p = cam_origin + ray_world * t;
                float tsdf = getTSDF(p);

                if (tsdf < EMPTY_TSDF) { // Probable geometry region
                    if (prev_tsdf > 0.0f && tsdf <= 0.0f) {
                        // Surface zero-crossing found
                        float t_hit = t - vs * tsdf / (tsdf - prev_tsdf + 1e-6f);
                        Eigen::Vector3f hit_world = cam_origin + ray_world * t_hit;
                        
                        vertices_out[out_idx] = hit_world;
                        normals_out[out_idx]  = computeNormal(hit_world);
                        
                        if (colors_out) {
                            Eigen::Vector3i vi = worldToVoxel(hit_world);
                            if (inBounds(vi.x(), vi.y(), vi.z())) {
                                const Voxel& vox = voxels_[idx(vi.x(), vi.y(), vi.z())];
                                colors_out[out_idx*3+0] = vox.r;
                                colors_out[out_idx*3+1] = vox.g;
                                colors_out[out_idx*3+2] = vox.b;
                            }
                        }
                        break;
                    }
                    prev_tsdf = tsdf;
                    t += vs * 0.5f; // Small steps near surface
                } else {
                    // Unknown or empty space
                    t += vs; 
                    prev_tsdf = EMPTY_TSDF;
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
    return voxels_[idx(x, y, z)];
}

Voxel& TSDFVolume::voxelAt(int x, int y, int z) {
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
                        local_points.push_back(voxelToWorld(x, y, z));
                        local_colors.push_back(vox.r);
                        local_colors.push_back(vox.g);
                        local_colors.push_back(vox.b);
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
