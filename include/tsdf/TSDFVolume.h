#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#ifdef CUDA_ENABLED
#include "utils/CudaUniquePtr.h"
#endif
#ifdef HIP_ENABLED
#include "utils/HipUniquePtr.h"
#include <hip/hip_vector_types.h>
#endif
#include "tsdf/VoxelGPU.h"

namespace kfusion {
namespace tsdf {

// Canonical CPU empty-voxel state (docs/CANONICAL_SEMANTICS.md, "TSDF volume").
// Emptiness is decided by weight, never by the tsdf literal: EMPTY_TSDF is what an
// unobserved voxel carries ("free space, not yet truncated"), deliberately NOT the
// zero isosurface. Named so the reset fill and the unobserved sampler value cannot
// drift apart. CUDA/HIP adopt them only via the deferred backend migration.
inline constexpr float EMPTY_TSDF   = 1.0f;
inline constexpr float EMPTY_WEIGHT = 0.0f;

// Color an unobserved voxel carries (no integration has written it yet). The CPU
// voxel color domain is float sRGB [0,1] (Todo 19), so the neutral byte 128 is
// written as the exact quotient 128.0f / 255.0f. A raw 120 in this field is not a
// color: it is an out-of-range sRGB component, and the extraction boundary CLAMPS
// finite values, so a raw 120 publishes byte 255 rather than being refused. Only a
// non-finite color is rejected (see kfusion::utils::srgbFloatToUint8). This value
// round-trips through that helper back to byte 128, which is the neutral byte the
// reset fill and every extraction boundary publish for an unobserved voxel.
inline constexpr float EMPTY_COLOR = 128.0f / 255.0f;

struct TSDFParams {
    int   resolution     = 256;          // 256³ voxels
    float voxel_size     = 0.010f;       // meters per voxel (256 * 0.010 = 2.56m) — covers full Kinect range
    float truncation     = 0.030f;       // meters (3 voxels at 0.010m)
    float max_weight     = 128.0f;
    Eigen::Vector3f origin = {-1.28f, -1.28f, 0.0f}; // Z: [0, 2.56m], XY: [-1.28, +1.28m]
    // Active depth band, camera-plane Z-depth in meters. raycast() bounds its march from
    // these two fields. integrate()/integrateCPU() do NOT read them: they gate on the
    // min_depth/max_depth arguments given at that public entry point. So this pair is the
    // raycast bound plus part of the parameter identity, mirrored from the
    // FusionHyperparams owner by syncTsdfDepthFromRange(). Neither kernel carries a
    // hard-coded literal. A band change differs under sameParams(), so it clears the volume.
    float min_depth      = 0.30f;        // nearest measured surface the raycast trusts
    float max_depth      = 5.00f;        // farthest measured surface the raycast trusts
};

struct Voxel {
    float tsdf   = EMPTY_TSDF;    // normalized [-1,1]
    float weight = EMPTY_WEIGHT;
    // sRGB-encoded [0,1], accumulated as float so repeated fusion converges to
    // the true mean instead of a per-update uint8 rounding dead zone. The byte
    // stage belongs to the extraction boundaries (utils::srgbFloatToUint8).
    float r = EMPTY_COLOR, g = EMPTY_COLOR, b = EMPTY_COLOR;
};

static_assert(Voxel{}.tsdf == EMPTY_TSDF && Voxel{}.weight == EMPTY_WEIGHT &&
                  Voxel{}.r == EMPTY_COLOR && Voxel{}.g == EMPTY_COLOR &&
                  Voxel{}.b == EMPTY_COLOR,
              "default-constructed Voxel must be the canonical empty state");

class TSDFVolume {
public:
    explicit TSDFVolume(const TSDFParams& params = TSDFParams{});
    ~TSDFVolume();

    // Non-copyable (large memory block)
    TSDFVolume(const TSDFVolume&) = delete;
    TSDFVolume& operator=(const TSDFVolume&) = delete;

    const TSDFParams& params() const { return params_; }

    /** Replace parameters. A resolution change reallocates voxel storage; ANY
        field difference clears the volume, so no voxel state derived from the
        previous geometry or fusion parameters survives a parameter change. */
    void setParams(const TSDFParams& p);

    // Integrate a depth frame into the volume
    // pose: world-from-camera 4x4 matrix
    // depth: meters per pixel (FRAME_W x FRAME_H)
    // rgb: optional (FRAME_W x FRAME_H * 3)
    void integrate(const float*            depth_meters,
                   const uint8_t*          rgb,
                   const Eigen::Matrix4f&  pose,
                   float                   fx, float fy,
                   float                   cx, float cy,
                   int                     width, int height,
                   float                   min_depth = 0.3f,
                   float                   max_depth = 5.0f);

    // Raycast the TSDF volume to produce a model frame
    void raycast(const Eigen::Matrix4f&  pose,
                 float fx, float fy, float cx, float cy,
                 int width, int height,
                 Eigen::Vector3f*        vertices_out,
                 Eigen::Vector3f*        normals_out,
                 uint8_t*                colors_out = nullptr) const;

    /** Extract all occupied voxels as a point cloud (for full model view). */
    void extractGlobalPointCloud(std::vector<Eigen::Vector3f>& points_out,
                                 std::vector<uint8_t>&         colors_out) const;

    // Reset all voxels to initial state
    void reset();

    // Get voxel at integer coordinates. PRECONDITION: x, y, z are all inside
    // [0, resolution) — callers consult inBounds()/their own loop bounds first.
    // The precondition is assert-checked in debug builds; release builds do the
    // bare indexed read on purpose (this sits in the meshing hot path).
    const Voxel& voxelAt(int x, int y, int z) const;
    Voxel&       voxelAt(int x, int y, int z);

    // Volume usage: fraction of voxels with weight > 0. Exact full scan, O(volume):
    // tests and offline use only, never per frame.
    float usageFraction() const;
    // O(1) fraction of voxels first observed by integrate() (CPU path) since the
    // last reset. Voxels written directly through voxelAt() are not counted.
    float observedFraction() const {
        return static_cast<float>(observed_voxels_.load(std::memory_order_relaxed)) /
               static_cast<float>(voxels_.size());
    }

    // integrated frame count
    int integratedFrames() const { return integrated_frames_.load(); }

    // Voxel data for mesh extraction (read-only)
    const std::vector<Voxel>& voxelData() const { return voxels_; }

    // Convert world position to voxel index
    Eigen::Vector3i worldToVoxel(const Eigen::Vector3f& world) const;
    Eigen::Vector3f voxelToWorld(const Eigen::Vector3i& v) const;
    Eigen::Vector3f voxelToWorld(int x, int y, int z) const;

#ifdef CUDA_ENABLED
    void initGPU();
    void freeGPU();
    void syncToGPU();
    void syncFromGPU();
    void integrateGPU(const float*           d_depth,
                      const uint8_t*         d_rgb,
                      const Eigen::Matrix4f& pose,
                      float fx, float fy, 
                      float cx, float cy,
                      int width, int height,
                      float min_depth = 0.3f,
                      float max_depth = 5.0f);
    void raycastGPU(const Eigen::Matrix4f& pose,
                    float fx, float fy, float cx, float cy,
                    int width, int height,
                    float3* d_vertices, float3* d_normals,
                    uchar3* d_colors = nullptr);

    void extractGlobalPointCloudGPU(std::vector<Eigen::Vector3f>& points_out,
                                    std::vector<uint8_t>&         colors_out) const;

    void* getGPUVoxels() const { return (void*)d_voxels_.get(); }
#elif defined(HIP_ENABLED)
    void initGPU();
    void freeGPU();
    void syncToGPU();
    void syncFromGPU();
    void integrateGPU(const float*           d_depth,
                      const uint8_t*         d_rgb,
                      const Eigen::Matrix4f& pose,
                      float fx, float fy, 
                      float cx, float cy,
                      int width, int height,
                      float min_depth = 0.3f,
                      float max_depth = 5.0f);
    void raycastGPU(const Eigen::Matrix4f& pose,
                    float fx, float fy, float cx, float cy,
                    int width, int height,
                    float3* d_vertices, float3* d_normals,
                    uchar3* d_colors = nullptr);

    void extractGlobalPointCloudGPU(std::vector<Eigen::Vector3f>& points_out,
                                    std::vector<uint8_t>&         colors_out) const;

    void* getGPUVoxels() const { return (void*)d_voxels_.get(); }
#endif

    void setGPUEnabled(bool enabled) { gpu_enabled_ = enabled; }
    bool isGPUEnabled() const { return gpu_enabled_; }

private:
    // Per-volume integration diagnostics (audit tsdf:T23: these counters used to
    // be one file-scope struct shared by every TSDFVolume in the process, so a
    // second volume silently polluted the first one's CSV rows). They live here
    // because this object is what actually produces them.
    //
    // Concurrency story: every write happens inside integrate(), i.e. while the
    // volume's own unique_lock is held — the integer counters that are updated
    // from inside an OpenMP region do so under `#pragma omp atomic` /
    // `#pragma omp critical`, the rest in the serial canonical fold. The single
    // read + reset is logDiagnostics(), also under that unique_lock, after the
    // parallel region's implicit barrier. There is deliberately no public
    // accessor: reading these without the volume lock would be a data race, and
    // the AZU_TSDF_LOG=1 CSV is the only consumer by design.
    struct Stats {
        int    depth_filtered      = 0;
        int    voxels_updated      = 0;   // contributing samples of sum_abs_sdf
        int    color_updates       = 0;
        int    truncation_clamped  = 0;
        float  max_abs_sdf         = 0.0f;
        double sum_abs_sdf         = 0.0; // sum over exactly voxels_updated samples
    };

    // Writes the AZU_TSDF_LOG CSV row for the current frame (avg computed as
    // sum_abs_sdf / voxels_updated, the contributing-sample count) and resets
    // stats_. Caller holds the unique_lock; no-op when the channel is off.
    void logDiagnostics();

    TSDFParams           params_;
    std::vector<Voxel>   voxels_;
    std::atomic<int>     integrated_frames_{0};
    std::atomic<int64_t> observed_voxels_{0};
    Stats                stats_;
    int                  stats_frame_{0};
    mutable std::shared_mutex mutex_;

    inline int idx(int x, int y, int z) const {
        return z * params_.resolution * params_.resolution
             + y * params_.resolution
             + x;
    }

    bool inBounds(int x, int y, int z) const {
        return x >= 0 && x < params_.resolution &&
               y >= 0 && y < params_.resolution &&
               z >= 0 && z < params_.resolution;
    }

    // CPU integration kernel
    void integrateCPU(const float* depth_meters,
                      const uint8_t* rgb,
                      const Eigen::Matrix4f& pose,
                      float fx, float fy, float cx, float cy,
                      int width, int height,
                      float min_depth, float max_depth);

    // Non-locking reset used by setParams() which already holds the unique_lock.
    void unlocked_reset();

#ifdef CUDA_ENABLED
    // GPU state
    utils::CudaUniquePtr<VoxelGPU> d_voxels_; 
    
    // Cached buffers for point cloud extraction
    mutable utils::CudaUniquePtr<uint32_t> d_pc_is_valid_;
    mutable utils::CudaUniquePtr<uint32_t> d_pc_offsets_;
    mutable utils::CudaUniquePtr<float3> d_pc_out_points_;
    mutable utils::CudaUniquePtr<uchar3> d_pc_out_colors_;
    
    // Cached buffers for safe thread-decoupled integration
    mutable utils::CudaUniquePtr<float> d_depth_integ_;
    mutable utils::CudaUniquePtr<uint8_t> d_rgb_integ_;
    mutable size_t last_depth_size_ = 0;
    mutable size_t last_rgb_size_ = 0;
    
    bool    gpu_valid_ = false;
#elif defined(HIP_ENABLED)
    // GPU state
    utils::HipUniquePtr<VoxelGPU> d_voxels_; 
    
    // Cached buffers for point cloud extraction
    mutable utils::HipUniquePtr<uint32_t> d_pc_is_valid_;
    mutable utils::HipUniquePtr<uint32_t> d_pc_offsets_;
    mutable utils::HipUniquePtr<float3> d_pc_out_points_;
    mutable utils::HipUniquePtr<uchar3> d_pc_out_colors_;
    
    // Cached buffers for safe thread-decoupled integration
    mutable utils::HipUniquePtr<float> d_depth_integ_;
    mutable utils::HipUniquePtr<uint8_t> d_rgb_integ_;
    mutable size_t last_depth_size_ = 0;
    mutable size_t last_rgb_size_ = 0;
    
    bool    gpu_valid_ = false;
#endif
    bool    gpu_enabled_ = false;

    // Internal helpers
    // Trilinear sample; false unless all 8 cell corners are observed (weight > 0).
    bool sampleTSDF(const Eigen::Vector3f& world_pos, float& f_out) const;
    // Unit gradient from 6 valid samples at +-voxel_size; false if any is invalid.
    bool computeNormal(const Eigen::Vector3f& world_pos, Eigen::Vector3f& n_out) const;
};

} // namespace tsdf
} // namespace kfusion
