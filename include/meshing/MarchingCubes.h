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

using ProgressCallback = std::function<void(float)>; // 0..1

class MarchingCubes {
public:
    MarchingCubes();
    ~MarchingCubes();

    // CPU extraction path
    std::shared_ptr<MeshData> extract(const tsdf::TSDFVolume& volume,
                                      ProgressCallback        cb = nullptr);

    // GPU extraction path
    std::shared_ptr<MeshData> extractGPU(const tsdf::TSDFVolume& volume);

    // Look-up tables for Marching Cubes moved to the shared CPU source of
    // truth: include/meshing/MarchingCubesTables.h (kfusion::meshing::tables).

private:
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

    static Eigen::Vector3f interpolateEdge(
        const Eigen::Vector3f& p1, float v1,
        const Eigen::Vector3f& p2, float v2);

    static Eigen::Vector3f computeNormal(
        const tsdf::TSDFVolume& vol, int x, int y, int z);
};

} // namespace meshing
} // namespace kfusion
