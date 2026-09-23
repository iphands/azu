#ifdef CUDA_ENABLED

#include "meshing/MarchingCubes.h"
#include "meshing/MarchingCubesTables.h"
#include "tsdf/VoxelGPU.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include "tsdf/VoxelGPU.h"
#include "utils/CudaUniquePtr.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

namespace kfusion {
namespace meshing {

// Lookup tables come from the one shared source, meshing/MarchingCubesTables.h
// (the CUDA file used to carry its own copy, including the corrupted
// edge_table[213..215] rows the CPU copy had already fixed).

// ---------------------------------------------------------------------------
// Constant Tables (Copied to Device)
// ---------------------------------------------------------------------------
__constant__ int c_edge_table[256];
__constant__ int c_tri_table[256][16];
__constant__ int c_corner_offsets[8][3];
__constant__ int c_edge_corners[12][2];

// ---------------------------------------------------------------------------
// GPU Kernels
// ---------------------------------------------------------------------------

__device__ float3 interpolateEdgeGPU(float3 p1, float v1, float3 p2, float v2) {
    if (fabsf(v1) < 1e-6f) return p1;
    if (fabsf(v2) < 1e-6f) return p2;
    float diff = v1 - v2;
    if (fabsf(diff) < 1e-6f) return p1;
    float t = fminf(fmaxf(v1 / diff, 0.0f), 1.0f);
    return make_float3(p1.x + t * (p2.x - p1.x),
                        p1.y + t * (p2.y - p1.y),
                        p1.z + t * (p2.z - p1.z));
}

// Compute normal via central difference on GPU
__device__ float3 computeNormalGPU(void* voxels_void, int resolution, int x, int y, int z) {
    tsdf::VoxelGPU* voxels = (tsdf::VoxelGPU*)voxels_void;
    
    auto sample = [&](int xi, int yi, int zi) {
        if (xi < 0 || xi >= resolution || yi < 0 || yi >= resolution || zi < 0 || zi >= resolution)
            return 1.0f;
        tsdf::VoxelGPU& v = voxels[zi * resolution * resolution + yi * resolution + xi];
        return (v.weight <= 0.001f) ? 1.0f : v.tsdf;
    };
    
    float dx = sample(x+1, y, z) - sample(x-1, y, z);
    float dy = sample(x, y+1, z) - sample(x, y-1, z);
    float dz = sample(x, y, z+1) - sample(x, y, z-1);
    float rlen = 1.0f / sqrtf(dx*dx + dy*dy + dz*dz + 1e-9f);
    return make_float3(dx * rlen, dy * rlen, dz * rlen);
}

__global__ void classifyVoxelKernel(
    void* voxels_void, int resolution, uint32_t* tri_counts)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution - 1 || y >= resolution - 1 || z >= resolution - 1) return;

    tsdf::VoxelGPU* voxels = (tsdf::VoxelGPU*)voxels_void;

    bool valid_cube = true;
    int cube_idx = 0;
    for (int i = 0; i < 8; ++i) {
        tsdf::VoxelGPU& v = voxels[(z + c_corner_offsets[i][2]) * resolution * resolution + 
                                   (y + c_corner_offsets[i][1]) * resolution + 
                                   (x + c_corner_offsets[i][0])];
        if (v.weight <= 0.001f) {
            valid_cube = false;
            break;
        }
        if (v.tsdf < 0) cube_idx |= (1 << i);
    }
    if (!valid_cube) cube_idx = 0;

    int tri_count = 0;
    if (c_edge_table[cube_idx] != 0) {
        for (int i = 0; i < 16 && c_tri_table[cube_idx][i] != -1; i += 3) {
            tri_count++;
        }
    }
    tri_counts[z * resolution * resolution + y * resolution + x] = tri_count;
}

__global__ void generateMeshKernel(
    void* voxels_void, int resolution, float voxel_size, float3 origin,
    const uint32_t* offsets, size_t max_tris,
    float3* out_v, float3* out_n, uint8_t* out_c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution - 1 || y >= resolution - 1 || z >= resolution - 1) return;

    int idx = z * resolution * resolution + y * resolution + x;
    uint32_t offset = offsets[idx];
    uint32_t next_offset = offsets[idx + 1];
    if (offset == next_offset) return;
    
    // Drop the strict next_offset > max_tris check to allow partial emission of triangles
    if (offset >= max_tris) return;

    tsdf::VoxelGPU* voxels = (tsdf::VoxelGPU*)voxels_void;

    auto get_voxel = [&](int xi, int yi, int zi) -> tsdf::VoxelGPU& {
        return voxels[zi * resolution * resolution + yi * resolution + xi];
    };

    bool valid_cube = true;
    int cube_idx = 0;
    float corner_vals[8];
    for (int i = 0; i < 8; ++i) {
        tsdf::VoxelGPU& v = get_voxel(x + c_corner_offsets[i][0], y + c_corner_offsets[i][1], z + c_corner_offsets[i][2]);
        if (v.weight <= 0.001f) valid_cube = false;
        corner_vals[i] = v.tsdf;
        if (corner_vals[i] < 0) cube_idx |= (1 << i);
    }
    if (!valid_cube) return;

    float3 edge_v[12];
    float3 edge_n[12];
    uchar3 edge_c[12];
    for (int i = 0; i < 12; ++i) {
        if (c_edge_table[cube_idx] & (1 << i)) {
            int c1 = c_edge_corners[i][0];
            int c2 = c_edge_corners[i][1];
            float3 p1 = make_float3(origin.x + (x + c_corner_offsets[c1][0]) * voxel_size,
                                    origin.y + (y + c_corner_offsets[c1][1]) * voxel_size,
                                    origin.z + (z + c_corner_offsets[c1][2]) * voxel_size);
            float3 p2 = make_float3(origin.x + (x + c_corner_offsets[c2][0]) * voxel_size,
                                    origin.y + (y + c_corner_offsets[c2][1]) * voxel_size,
                                    origin.z + (z + c_corner_offsets[c2][2]) * voxel_size);
            edge_v[i] = interpolateEdgeGPU(p1, corner_vals[c1], p2, corner_vals[c2]);
            if (isnan(edge_v[i].x) || isnan(edge_v[i].y) || isnan(edge_v[i].z)) {
                edge_v[i] = make_float3(0,0,0);
            }
            
            float3 n1 = computeNormalGPU(voxels_void, resolution, x+c_corner_offsets[c1][0], y+c_corner_offsets[c1][1], z+c_corner_offsets[c1][2]);
            float3 n2 = computeNormalGPU(voxels_void, resolution, x+c_corner_offsets[c2][0], y+c_corner_offsets[c2][1], z+c_corner_offsets[c2][2]);
            
            float diff = corner_vals[c1] - corner_vals[c2];
            float t = (fabsf(diff) < 1e-6f) ? 0.0f : fminf(fmaxf(corner_vals[c1] / diff, 0.0f), 1.0f);
            
            edge_n[i] = make_float3(n1.x + t*(n2.x-n1.x), n1.y + t*(n2.y-n1.y), n1.z + t*(n2.z-n1.z));
            float len = 1.0f / sqrtf(edge_n[i].x*edge_n[i].x + edge_n[i].y*edge_n[i].y + edge_n[i].z*edge_n[i].z + 1e-9f);
            edge_n[i].x *= len; edge_n[i].y *= len; edge_n[i].z *= len;
            
            tsdf::VoxelGPU& v1 = get_voxel(x+c_corner_offsets[c1][0], y+c_corner_offsets[c1][1], z+c_corner_offsets[c1][2]);
            tsdf::VoxelGPU& v2 = get_voxel(x+c_corner_offsets[c2][0], y+c_corner_offsets[c2][1], z+c_corner_offsets[c2][2]);
            // Device voxel colour is sRGB in [0,255]; round to nearest like the
            // CPU's srgbFloatToUint8 (truncation darkened every channel).
            edge_c[i] = make_uchar3(
                (uint8_t)__float2int_rn(fminf(255.0f, fmaxf(0.0f, v1.r + t * (v2.r - v1.r)))),
                (uint8_t)__float2int_rn(fminf(255.0f, fmaxf(0.0f, v1.g + t * (v2.g - v1.g)))),
                (uint8_t)__float2int_rn(fminf(255.0f, fmaxf(0.0f, v1.b + t * (v2.b - v1.b))))
            );
        }
    }

    uint32_t out_idx_start = offset * 3;
    int v_count = 0;
    int current_tri = offset;
    for (int i = 0; i < 16 && c_tri_table[cube_idx][i] != -1; i += 3) {
        if (current_tri >= max_tris) break;
        for (int k = 0; k < 3; ++k) {
            // Reverse table order: the CPU-canonical outward winding (see
            // src/meshing/MarchingCubes.cpp and docs/CANONICAL_SEMANTICS.md).
            // Forward order made every GPU triangle face inward.
            int e = c_tri_table[cube_idx][i + 2 - k];
            out_v[out_idx_start + v_count] = edge_v[e];
            out_n[out_idx_start + v_count] = edge_n[e];
            out_c[(out_idx_start + v_count)*3+0] = edge_c[e].x;
            out_c[(out_idx_start + v_count)*3+1] = edge_c[e].y;
            out_c[(out_idx_start + v_count)*3+2] = edge_c[e].z;
            v_count++;
        }
        current_tri++;
    }
}

// ---------------------------------------------------------------------------
// Host Logic
// ---------------------------------------------------------------------------

void MarchingCubes::initGPU(int res) {
    if (res == last_resolution_ && d_voxel_tri_counts_) return;
    freeGPU();

    size_t n = (size_t)res * res * res;
    d_voxel_tri_counts_ = utils::make_cuda_unique<uint32_t>(n);
    d_voxel_offsets_    = utils::make_cuda_unique<uint32_t>(n + 1);
    
    d_mesh_vertices_ = utils::make_cuda_unique<float3>(max_triangles_ * 3);
    d_mesh_normals_  = utils::make_cuda_unique<float3>(max_triangles_ * 3);
    d_mesh_colors_   = utils::make_cuda_unique<uint8_t>(max_triangles_ * 3 * 3);

    // Tables from MarchingCubes.cpp
    const int corner_offsets[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const int edge_corners[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};

    const cudaError_t errs[4] = {
        cudaMemcpyToSymbol(c_edge_table, tables::edge_table, 256 * sizeof(int)),
        cudaMemcpyToSymbol(c_tri_table, tables::tri_table, 256 * 16 * sizeof(int)),
        cudaMemcpyToSymbol(c_corner_offsets, corner_offsets, 8 * 3 * sizeof(int)),
        cudaMemcpyToSymbol(c_edge_corners, edge_corners, 12 * 2 * sizeof(int)),
    };
    for (cudaError_t e : errs) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("MC table upload failed: ") + cudaGetErrorString(e));
        }
    }
    
    last_resolution_ = res;
}

void MarchingCubes::freeGPU() {
    d_voxel_tri_counts_.reset();
    d_voxel_offsets_.reset();
    d_mesh_vertices_.reset();
    d_mesh_normals_.reset();
    d_mesh_colors_.reset();
}

std::shared_ptr<MeshData> MarchingCubes::extractGPU(const tsdf::TSDFVolume& volume) {
    const auto& params = volume.params();
    initGPU(params.resolution);

    size_t n = (size_t)params.resolution * params.resolution * params.resolution;
    dim3 block(8, 8, 8);
    dim3 grid((params.resolution + 7)/8, (params.resolution + 7)/8, (params.resolution + 7)/8);

    // Initialize counts to zero to prevent uninitialized memory corruption from padding voxels
    CUDA_CHECK(cudaMemset(d_voxel_tri_counts_.get(), 0, n * sizeof(uint32_t)));

    classifyVoxelKernel<<<grid, block>>>((void*)volume.getGPUVoxels(), params.resolution, d_voxel_tri_counts_.get());
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    thrust::device_ptr<uint32_t> d_counts(d_voxel_tri_counts_.get());
    thrust::device_ptr<uint32_t> d_offsets(d_voxel_offsets_.get());
    thrust::exclusive_scan(d_counts, d_counts + n, d_offsets);
    
    uint32_t total_tris;
    CUDA_CHECK(cudaMemcpy(&total_tris, d_voxel_offsets_.get() + n - 1, 4, cudaMemcpyDeviceToHost));
    uint32_t last_count;
    CUDA_CHECK(cudaMemcpy(&last_count, d_voxel_tri_counts_.get() + n - 1, 4, cudaMemcpyDeviceToHost));
    total_tris += last_count;
    
    CUDA_CHECK(cudaMemcpy(d_voxel_offsets_.get() + n, &total_tris, 4, cudaMemcpyHostToDevice));

    std::shared_ptr<MeshData> mesh = std::make_shared<MeshData>();
    if (total_tris > 0) {
        uint32_t capped_tris = (total_tris > max_triangles_) ? (uint32_t)max_triangles_ : total_tris;
        
        float3 origin = {params.origin.x(), params.origin.y(), params.origin.z()};
        generateMeshKernel<<<grid, block>>>(
            (void*)volume.getGPUVoxels(), params.resolution, params.voxel_size, origin,
            d_voxel_offsets_.get(), max_triangles_, d_mesh_vertices_.get(), d_mesh_normals_.get(), d_mesh_colors_.get()
        );
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float3> raw_pos(capped_tris * 3);
        std::vector<float3> raw_norm(capped_tris * 3);
        std::vector<uint8_t> raw_col(capped_tris * 3 * 3);
        
        CUDA_CHECK(cudaMemcpy(raw_pos.data(), d_mesh_vertices_.get(), capped_tris * 3 * sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(raw_norm.data(), d_mesh_normals_.get(), capped_tris * 3 * sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(raw_col.data(), d_mesh_colors_.get(), capped_tris * 9, cudaMemcpyDeviceToHost));
        
        // Unify vertices on CPU to save memory (reduces size by ~6x)
        // Use spatial quantization to ensure vertices at slice boundaries are unified
        struct QuantizedPos {
            int ix, iy, iz;
            
            QuantizedPos(const Eigen::Vector3f& pos, float voxel_size) {
                ix = static_cast<int>(std::round(pos.x() / (voxel_size * 0.01f)));
                iy = static_cast<int>(std::round(pos.y() / (voxel_size * 0.01f)));
                iz = static_cast<int>(std::round(pos.z() / (voxel_size * 0.01f)));
            }
            
            bool operator==(const QuantizedPos& other) const {
                return ix == other.ix && iy == other.iy && iz == other.iz;
            }
        };
        
        struct QuantizedHash {
            size_t operator()(const QuantizedPos& q) const {
                return std::hash<int>{}(q.ix) ^ 
                       (std::hash<int>{}(q.iy) << 1) ^ 
                       (std::hash<int>{}(q.iz) << 2);
            }
        };
        
        std::unordered_map<QuantizedPos, uint32_t, QuantizedHash> global_map;
        
        mesh->positions.reserve(capped_tris);
        mesh->normals.reserve(capped_tris);
        mesh->colors.reserve(capped_tris * 3);
        mesh->indices.reserve(capped_tris * 3);

        for (uint32_t i = 0; i < capped_tris * 3; ++i) {
            Eigen::Vector3f pos(raw_pos[i].x, raw_pos[i].y, raw_pos[i].z);
            QuantizedPos qpos(pos, params.voxel_size);
            auto it = global_map.find(qpos);
            if (it != global_map.end()) {
                mesh->indices.push_back(it->second);
            } else {
                uint32_t new_idx = static_cast<uint32_t>(mesh->positions.size());
                global_map[qpos] = new_idx;
                mesh->positions.push_back(pos);
                mesh->normals.push_back(Eigen::Vector3f(raw_norm[i].x, raw_norm[i].y, raw_norm[i].z));
                mesh->colors.push_back(raw_col[i*3+0]);
                mesh->colors.push_back(raw_col[i*3+1]);
                mesh->colors.push_back(raw_col[i*3+2]);
                mesh->indices.push_back(new_idx);
            }
        }
        
        if (total_tris > max_triangles_) {
            std::cerr << "[MC] Reached max_triangles limit (" << max_triangles_ << "). Mesh is truncated.\n";
        }
    }

    return mesh;
}

} // namespace meshing
} // namespace kfusion

#endif // CUDA_ENABLED
