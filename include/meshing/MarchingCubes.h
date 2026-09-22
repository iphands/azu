#pragma once

#include "meshing/MeshData.h"
#include "tsdf/TSDFVolume.h"
#ifdef CUDA_ENABLED
#include "utils/CudaUniquePtr.h"
#endif
#ifdef HIP_ENABLED
#include "utils/HipUniquePtr.h"
#include <hip/hip_vector_types.h>
#endif
#include <functional>

namespace kfusion {
namespace meshing {

/** Extraction progress callback, argument in [0, 1].
 *
 * CPU contract (big-fix Todo 17): `extract()` invokes the callback ONLY from the
 * thread that called it, ONLY during the serial post-processing (weld) phase, and
 * never from inside the OpenMP extraction region. Consequences that a caller may
 * rely on and that `mesh_truncation_progress_contract` locks:
 *   - callbacks can never overlap (single-threaded, synchronous calls), so a
 *     non-reentrant callback needs no lock;
 *   - the emitted sequence is a pure function of the volume (same values in the
 *     same order on every run, independent of the OpenMP thread count);
 *   - every value is finite and within [0, 1], nondecreasing, and the final call
 *     is exactly 1.0.
 * The parallel extraction phase deliberately reports no progress, so the callback
 * is a merge-phase indicator rather than a live completion estimate. CUDA/HIP do
 * not use this callback; their deferral is documented in
 * docs/CUDA_HIP_DEFERRED_CHANGES.md. */
using ProgressCallback = std::function<void(float)>; // 0..1

class MarchingCubes {
public:
    MarchingCubes();
    ~MarchingCubes();

    // CPU extraction path
    std::shared_ptr<MeshData> extract(const tsdf::TSDFVolume& volume,
                                       ProgressCallback        cb = nullptr);

    // CPU triangle budget (big-fix Todo 17). `extract()` stops starting a new
    // triangle once this many rows are already emitted, sets MeshData::truncated,
    // and never emits a partial triangle. Default matches the GPU `max_triangles_`
    // budget so CPU and the (deferred) GPU backends share one cap value; it is a
    // CPU-side field and touches no CUDA/HIP buffer or kernel.
    void   setMaxTriangles(size_t max_triangles) { max_triangles_cpu_ = max_triangles; }
    size_t maxTriangles() const { return max_triangles_cpu_; }

    // GPU extraction path
    std::shared_ptr<MeshData> extractGPU(const tsdf::TSDFVolume& volume);

    // Look-up tables for Marching Cubes moved to the shared CPU source of
    // truth: include/meshing/MarchingCubesTables.h (kfusion::meshing::tables).

private:
    // CPU triangle cap used by extract(); see setMaxTriangles(). Named apart from
    // the GPU `max_triangles_` below so adding it never collides with the backend
    // block that stays compiled out on the CPU lane.
    size_t max_triangles_cpu_ = 2000000;

#ifdef CUDA_ENABLED

    // Persistent GPU buffers owned by this instance
    utils::CudaUniquePtr<uint32_t> d_voxel_tri_counts_;
    utils::CudaUniquePtr<uint32_t> d_voxel_offsets_;
    utils::CudaUniquePtr<float3>   d_mesh_vertices_;
    utils::CudaUniquePtr<float3>   d_mesh_normals_;
    utils::CudaUniquePtr<uint8_t>  d_mesh_colors_;
    
    size_t    max_triangles_      = 2000000;
    int       last_resolution_    = 0;

    void initGPU(int resolution);
    void freeGPU();
#elif defined(HIP_ENABLED)
    // Persistent GPU buffers owned by this instance
    utils::HipUniquePtr<uint32_t> d_voxel_tri_counts_;
    utils::HipUniquePtr<uint32_t> d_voxel_offsets_;
    utils::HipUniquePtr<float3>   d_mesh_vertices_;
    utils::HipUniquePtr<float3>   d_mesh_normals_;
    utils::HipUniquePtr<uint8_t>  d_mesh_colors_;
    
    size_t    max_triangles_      = 2000000;
    int       last_resolution_    = 0;

    void initGPU(int resolution);
    void freeGPU();
#endif

    // One canonical zero-crossing of an edge, carrying the single interpolation
    // parameter `t` alongside the position it produced. Returning `t` in the same
    // value is what lets position, normal and color consume ONE parameter (a
    // position-only return structurally cannot share it - the old interpolateEdge
    // defect, meshing:D8). `ok` is false for any non-finite input or collapsed
    // blend so the caller refuses the edge rather than substituting.
    struct EdgeCrossing {
        Eigen::Vector3f position;
        float           t;
        bool            ok;
    };

    // Parameterise a crossing from its canonical LOWER endpoint (v_low at p_low)
    // toward the UPPER endpoint (v_up at p_up). Both endpoints must be supported and
    // finite. `t` is measured from the lower endpoint so the same physical edge
    // yields the identical `t` and payload no matter which adjacent cube reached it.
    static EdgeCrossing crossingFromLower(const Eigen::Vector3f& p_low, float v_low,
                                          const Eigen::Vector3f& p_up,  float v_up);

    /** The one CPU corner/volume normal primitive: the unit TSDF gradient at
        voxel (x,y,z), sampled only from in-bounds voxels that are observed and
        finite. Where a two-sided difference is unavailable - volume border, an
        unobserved or non-finite neighbour - that axis falls back to a one-sided
        difference at the same per-voxel-step scale; an out-of-bounds or
        unobserved sample is never substituted into the difference. Returns a
        NaN-filled vector when no usable gradient exists, so the caller refuses
        the vertex instead of emitting a fabricated normal. */
    static Eigen::Vector3f voxelNormal(const tsdf::TSDFVolume& vol, int x, int y, int z);
};

} // namespace meshing
} // namespace kfusion
