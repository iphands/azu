#include "meshing/MarchingCubes.h"
#include "meshing/MarchingCubesTables.h"
#include <cmath>
#include <array>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <unordered_map>

namespace kfusion {
namespace meshing {

// Helper for vertex unification
struct VectorHash {
    size_t operator()(const Eigen::Vector3f& v) const {
        size_t h1 = std::hash<float>{}(v.x());
        size_t h2 = std::hash<float>{}(v.y());
        size_t h3 = std::hash<float>{}(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

MarchingCubes::MarchingCubes() {}
MarchingCubes::~MarchingCubes() {
#ifdef CUDA_ENABLED
    freeGPU();
#elif defined(HIP_ENABLED)
    freeGPU();
#endif
}

// ---------------------------------------------------------------------------
// Lookup tables (edge_table / tri_table) live in the single shared CPU source
// of truth: include/meshing/MarchingCubesTables.h
// (kfusion::meshing::tables::{edge_table, tri_table}). Values are unchanged
// from the former class-static definitions; the known corrupt
// edge_table[213/214/215] entries are owned by big-fix Todos 6-7, not fixed here.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Edge vertex positions and vertex 0..7 offsets for a cube
// ---------------------------------------------------------------------------

// Corner offsets relative to voxel (x,y,z)
static const int CORNER_OFFSETS[8][3] = {
    {0,0,0},{1,0,0},{1,1,0},{0,1,0},
    {0,0,1},{1,0,1},{1,1,1},{0,1,1}
};

// Edge → pair of corner indices
static const int EDGE_CORNERS[12][2] = {
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
};

Eigen::Vector3f MarchingCubes::interpolateEdge(
    const Eigen::Vector3f& p1, float v1,
    const Eigen::Vector3f& p2, float v2)
{
    if (std::abs(v1) < 1e-6f) return p1;
    if (std::abs(v2) < 1e-6f) return p2;
    float diff = v1 - v2;
    if (std::abs(diff) < 1e-6f) return p1;
    float t = std::max(0.0f, std::min(1.0f, v1 / diff));
    return p1 + t * (p2 - p1);
}

Eigen::Vector3f MarchingCubes::computeNormal(
    const tsdf::TSDFVolume& vol, int x, int y, int z)
{
    auto tsdf_safe = [&](int xi, int yi, int zi) -> float {
        const auto& p = vol.params();
        if (xi < 0 || xi >= p.resolution ||
            yi < 0 || yi >= p.resolution ||
            zi < 0 || zi >= p.resolution)
            return 1.0f;
        return vol.voxelAt(xi, yi, zi).tsdf;
    };

    float dx = tsdf_safe(x+1,y,z) - tsdf_safe(x-1,y,z);
    float dy = tsdf_safe(x,y+1,z) - tsdf_safe(x,y-1,z);
    float dz = tsdf_safe(x,y,z+1) - tsdf_safe(x,y,z-1);
    Eigen::Vector3f n(dx, dy, dz);
    float len = n.norm();
    return (len > 1e-6f) ? (n / len) : Eigen::Vector3f(0,0,1);
}

#include <unordered_map>

// Helper to hash Eigen vectors for vertex unification
struct VectorHasher {
    size_t operator()(const Eigen::Vector3f& v) const {
        size_t h1 = std::hash<float>{}(v.x());
        size_t h2 = std::hash<float>{}(v.y());
        size_t h3 = std::hash<float>{}(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::shared_ptr<MeshData> MarchingCubes::extract(const tsdf::TSDFVolume& volume,
                                            ProgressCallback progress_cb)
{
    const auto& p     = volume.params();
    const int   RES_X = p.resolution;
    const int   RES_Y = p.resolution;
    const int   RES_Z = p.resolution;

    std::shared_ptr<MeshData> mesh_final = std::make_shared<MeshData>();

    // For vertex unification and smooth normals
    // We'll use a thread-local approach to avoid contention during extraction, 
    // then merge at the end. Or just use face vertices for now but indexed.
    // Actually, to truly satisfy "Smooth Shading", we MUST use vertex normals.
    
    // Structure to hold unique vertex data
    struct Vertex {
        Eigen::Vector3f pos;
        Eigen::Vector3f norm;
        uint8_t color[3];

        bool operator==(const Vertex& other) const {
            return pos.isApprox(other.pos, 1e-4f);
        }
    };

    struct VertexHasher {
        size_t operator()(const Vertex& v) const {
            return std::hash<float>{}(v.pos.x()) ^ 
                   (std::hash<float>{}(v.pos.y()) << 1) ^ 
                   (std::hash<float>{}(v.pos.z()) << 2);
        }
    };

    // Parallelize over slices (Z direction)
    std::vector<MeshData> slice_meshes(RES_Z - 1);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int z = 0; z < RES_Z - 1; ++z) {

        if (progress_cb && (z % 10 == 0)) 
            progress_cb(static_cast<float>(z) / (RES_Z - 2));

        auto& local_mesh = slice_meshes[z];
        const float vs = p.voxel_size;
        
        for (int y = 0; y < RES_Y - 1; ++y) {
            for (int x = 0; x < RES_X - 1; ++x) {
                bool valid_cube = true;
                float corner_vals[8];
                for (int c = 0; c < 8; ++c) {
                    int cx = x + CORNER_OFFSETS[c][0];
                    int cy = y + CORNER_OFFSETS[c][1];
                    int cz = z + CORNER_OFFSETS[c][2];
                    const tsdf::Voxel& vox = volume.voxelAt(cx, cy, cz);
                    if (vox.weight <= 0.001f) valid_cube = false;
                    corner_vals[c] = vox.tsdf;
                }

                if (!valid_cube) continue;

                int cube_idx = 0;
                for (int c = 0; c < 8; ++c)
                    if (corner_vals[c] < 0.0f) cube_idx |= (1 << c);

                if (tables::edge_table[cube_idx] == 0) continue;

                Eigen::Vector3f corner_pos[8];
                for (int c = 0; c < 8; ++c) {
                    corner_pos[c] = volume.voxelToWorld(
                        x + CORNER_OFFSETS[c][0],
                        y + CORNER_OFFSETS[c][1],
                        z + CORNER_OFFSETS[c][2]);
                }

                Eigen::Vector3f edge_verts[12];
                Eigen::Vector3f edge_norms[12];
                uint8_t         edge_colors[12][3];

                for (int e = 0; e < 12; ++e) {
                    if (tables::edge_table[cube_idx] & (1 << e)) {
                        int c0 = EDGE_CORNERS[e][0];
                        int c1 = EDGE_CORNERS[e][1];
                        
                        const tsdf::Voxel& vox0 = volume.voxelAt(x + CORNER_OFFSETS[c0][0], y + CORNER_OFFSETS[c0][1], z + CORNER_OFFSETS[c0][2]);
                        const tsdf::Voxel& vox1 = volume.voxelAt(x + CORNER_OFFSETS[c1][0], y + CORNER_OFFSETS[c1][1], z + CORNER_OFFSETS[c1][2]);

                        edge_verts[e] = interpolateEdge(
                            corner_pos[c0], corner_vals[c0],
                            corner_pos[c1], corner_vals[c1]);
                        
                        Eigen::Vector3f n0 = computeNormal(volume, 
                            x + CORNER_OFFSETS[c0][0], 
                            y + CORNER_OFFSETS[c0][1], 
                            z + CORNER_OFFSETS[c0][2]);
                        Eigen::Vector3f n1 = computeNormal(volume, 
                            x + CORNER_OFFSETS[c1][0], 
                            y + CORNER_OFFSETS[c1][1], 
                            z + CORNER_OFFSETS[c1][2]);
                        
                        float diff = corner_vals[c0] - corner_vals[c1];
                        float t = (std::abs(diff) < 1e-6f) ? 0.0f : 
                                  (corner_vals[c0] / diff);
                        t = std::max(0.0f, std::min(1.0f, t));

                        edge_norms[e] = (n0 + t * (n1 - n0)).normalized();
                        
                        edge_colors[e][0] = static_cast<uint8_t>(vox0.r + t * (static_cast<float>(vox1.r) - vox0.r));
                        edge_colors[e][1] = static_cast<uint8_t>(vox0.g + t * (static_cast<float>(vox1.g) - vox0.g));
                        edge_colors[e][2] = static_cast<uint8_t>(vox0.b + t * (static_cast<float>(vox1.b) - vox0.b));
                    }
                }

                for (int t = 0; tables::tri_table[cube_idx][t] != -1; t += 3) {
                    // Reverse winding (2, 1, 0 instead of 0, 1, 2) to fix front-face culling
                    for (int i = 2; i >= 0; --i) {
                        int e = tables::tri_table[cube_idx][t + i];
                        uint32_t vidx = static_cast<uint32_t>(local_mesh.positions.size());
                        local_mesh.positions.push_back(edge_verts[e]);
                        // TSDF gradient already points OUT (toward camera), so keep it positive
                        local_mesh.normals.push_back(edge_norms[e]);
                        local_mesh.colors.push_back(edge_colors[e][0]);
                        local_mesh.colors.push_back(edge_colors[e][1]);
                        local_mesh.colors.push_back(edge_colors[e][2]);
                        local_mesh.indices.push_back(vidx);
                    }
                }
            }
        }
    }

    // Merge results with unified vertex mapping
    // Use spatial quantization to ensure vertices at slice boundaries are unified
    std::shared_ptr<MeshData> total_mesh = std::make_shared<MeshData>();
    
    // Quantize positions to voxel grid for reliable unification
    struct QuantizedPos {
        int ix, iy, iz;
        
        QuantizedPos(const Eigen::Vector3f& pos, float voxel_size) {
            ix = static_cast<int>(std::round(pos.x() / voxel_size));
            iy = static_cast<int>(std::round(pos.y() / voxel_size));
            iz = static_cast<int>(std::round(pos.z() / voxel_size));
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

    for (const auto& sm : slice_meshes) {
        if (sm.empty()) continue;
        
        for (size_t i = 0; i < sm.indices.size(); ++i) {
            uint32_t old_idx = sm.indices[i];
            const auto& pos = sm.positions[old_idx];
            
            QuantizedPos qpos(pos, p.voxel_size * 0.01f); // Fine quantization for accuracy
            auto it = global_map.find(qpos);
            if (it != global_map.end()) {
                total_mesh->indices.push_back(it->second);
            } else {
                uint32_t new_idx = static_cast<uint32_t>(total_mesh->positions.size());
                global_map[qpos] = new_idx;
                total_mesh->positions.push_back(pos);
                total_mesh->normals.push_back(sm.normals[old_idx]);
                if (sm.colors.size() == sm.positions.size() * 3) {
                    total_mesh->colors.push_back(sm.colors[old_idx*3+0]);
                    total_mesh->colors.push_back(sm.colors[old_idx*3+1]);
                    total_mesh->colors.push_back(sm.colors[old_idx*3+2]);
                }
                total_mesh->indices.push_back(new_idx);
            }
        }
    }

    if (progress_cb) progress_cb(1.0f);
    return total_mesh;
}

} // namespace meshing
} // namespace kfusion
