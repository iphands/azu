#include "meshing/MarchingCubes.h"
#include "meshing/MarchingCubesTables.h"
#include <cmath>
#include <array>
#include <iostream>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <unordered_map>

namespace kfusion {
namespace meshing {

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
// (kfusion::meshing::tables::{edge_table, tri_table}). big-fix Todo 7 repaired
// edge_table[213/214/215] there to the canonical crossing-edge values
// 0x83f / 0xb35 / 0xa3c; the CUDA/HIP duplicate copies remain corrupt and
// deferred (not compiled or runtime-tested on the CPU lane).
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

namespace {

// Canonical CPU observation threshold. A voxel at or below this weight has never
// been integrated, so it carries no measured surface: it samples as the canonical
// empty value EMPTY_TSDF (+1.0f, outside) whatever tsdf literal is in it, and it
// cannot support a crossing edge. Same epsilon the old cube-level gate used, now
// applied per corner instead of per cube.
constexpr float kWeightEpsilon   = 0.001f;
constexpr float kGradientEpsilon = 1e-6f;
constexpr float kInterpEpsilon   = 1e-6f;

bool isFiniteVec(const Eigen::Vector3f& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

Eigen::Vector3f invalidNormal() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    return Eigen::Vector3f(nan, nan, nan);
}

// Weight-guarded scalar-field sample plus its support bit (canonical rule R1).
struct CornerSample {
    float value;
    bool  supported;
};

CornerSample sampleCorner(const tsdf::Voxel& v) {
    if (!std::isfinite(v.tsdf) || v.weight <= kWeightEpsilon)
        return {tsdf::EMPTY_TSDF, false};
    return {v.tsdf, true};
}

} // namespace

Eigen::Vector3f MarchingCubes::interpolateEdge(
    const Eigen::Vector3f& p1, float v1,
    const Eigen::Vector3f& p2, float v2)
{
    if (!std::isfinite(v1) || !std::isfinite(v2) || !isFiniteVec(p1) || !isFiniteVec(p2))
        return invalidNormal();
    if (std::abs(v1) < kInterpEpsilon) return p1;
    if (std::abs(v2) < kInterpEpsilon) return p2;
    float diff = v1 - v2;
    if (!std::isfinite(diff) || std::abs(diff) < kInterpEpsilon) return p1;
    float t = std::max(0.0f, std::min(1.0f, v1 / diff));
    return p1 + t * (p2 - p1);
}

Eigen::Vector3f MarchingCubes::voxelNormal(const tsdf::TSDFVolume& vol, int x, int y, int z) {
    const auto& p = vol.params();
    const int   R = p.resolution;

    auto usable = [&](int xi, int yi, int zi, float& out) -> bool {
        if (xi < 0 || xi >= R || yi < 0 || yi >= R || zi < 0 || zi >= R) return false;
        const CornerSample s = sampleCorner(vol.voxelAt(xi, yi, zi));
        if (!s.supported) return false;
        out = s.value;
        return true;
    };

    float self = 0.0f;
    if (!usable(x, y, z, self)) return invalidNormal();

    Eigen::Vector3f grad(0.0f, 0.0f, 0.0f);
    for (int axis = 0; axis < 3; ++axis) {
        const int ox = (axis == 0) ? 1 : 0;
        const int oy = (axis == 1) ? 1 : 0;
        const int oz = (axis == 2) ? 1 : 0;
        float hi = 0.0f, lo = 0.0f;
        const bool have_hi = usable(x + ox, y + oy, z + oz, hi);
        const bool have_lo = usable(x - ox, y - oy, z - oz, lo);
        // Central differences span two voxel steps, so they carry the 0.5 factor;
        // a one-sided fallback spans one. Without it the border axis would be
        // weighted twice as heavily as an interior axis of the same physical slope.
        if (have_hi && have_lo)   grad[axis] = 0.5f * (hi - lo);
        else if (have_hi)         grad[axis] = hi - self;
        else if (have_lo)         grad[axis] = self - lo;
        else                      grad[axis] = 0.0f;
    }

    if (!isFiniteVec(grad)) return invalidNormal();
    const float len = grad.norm();
    if (!std::isfinite(len) || len <= kGradientEpsilon) return invalidNormal();
    return grad / len;
}

std::shared_ptr<MeshData> MarchingCubes::extract(const tsdf::TSDFVolume& volume,
                                            ProgressCallback progress_cb)
{
    const auto& p     = volume.params();
    const int   RES_X = p.resolution;
    const int   RES_Y = p.resolution;
    const int   RES_Z = p.resolution;

    std::shared_ptr<MeshData> mesh_final = std::make_shared<MeshData>();

    // Parallelize over slices (Z direction)
    std::vector<MeshData> slice_meshes(RES_Z - 1);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int z = 0; z < RES_Z - 1; ++z) {

        if (progress_cb && (z % 10 == 0)) 
            progress_cb(static_cast<float>(z) / (RES_Z - 2));

        auto& local_mesh = slice_meshes[z];
        
        for (int y = 0; y < RES_Y - 1; ++y) {
            for (int x = 0; x < RES_X - 1; ++x) {
                tsdf::Voxel corner_vox[8];
                CornerSample corner[8];
                for (int c = 0; c < 8; ++c) {
                    corner_vox[c] = volume.voxelAt(x + CORNER_OFFSETS[c][0],
                                                   y + CORNER_OFFSETS[c][1],
                                                   z + CORNER_OFFSETS[c][2]);
                    corner[c] = sampleCorner(corner_vox[c]);
                }

                int cube_idx = 0;
                for (int c = 0; c < 8; ++c)
                    if (corner[c].value < 0.0f) cube_idx |= (1 << c);

                const int edge_mask = tables::edge_table[cube_idx];
                if (edge_mask == 0) continue;

                Eigen::Vector3f corner_pos[8];
                for (int c = 0; c < 8; ++c) {
                    corner_pos[c] = volume.voxelToWorld(
                        x + CORNER_OFFSETS[c][0],
                        y + CORNER_OFFSETS[c][1],
                        z + CORNER_OFFSETS[c][2]);
                }

                // One gradient evaluation per cube corner, shared by every crossing
                // edge that touches it. The old path recomputed it per edge endpoint,
                // up to 24 evaluations for the same 8 corners.
                Eigen::Vector3f corner_norm[8];
                bool corner_norm_done[8] = {false, false, false, false,
                                            false, false, false, false};
                auto cornerNormalAt = [&](int c) -> Eigen::Vector3f {
                    if (!corner_norm_done[c]) {
                        corner_norm[c] = voxelNormal(volume,
                                                     x + CORNER_OFFSETS[c][0],
                                                     y + CORNER_OFFSETS[c][1],
                                                     z + CORNER_OFFSETS[c][2]);
                        corner_norm_done[c] = true;
                    }
                    return corner_norm[c];
                };

                Eigen::Vector3f edge_verts[12];
                Eigen::Vector3f edge_norms[12];
                uint8_t         edge_colors[12][3];
                bool            edge_ok[12] = {false, false, false, false,
                                               false, false, false, false,
                                               false, false, false, false};

                for (int e = 0; e < 12; ++e) {
                    if (!(edge_mask & (1 << e))) continue;
                    const int c0 = EDGE_CORNERS[e][0];
                    const int c1 = EDGE_CORNERS[e][1];

                    // A crossing edge is emitted only when both of its endpoints
                    // carry measured, finite data. An unobserved corner elsewhere in
                    // the cube no longer deletes the cube.
                    if (!corner[c0].supported || !corner[c1].supported) continue;

                    const Eigen::Vector3f vtx = interpolateEdge(
                        corner_pos[c0], corner[c0].value,
                        corner_pos[c1], corner[c1].value);
                    if (!isFiniteVec(vtx)) continue;

                    const Eigen::Vector3f n0 = cornerNormalAt(c0);
                    const Eigen::Vector3f n1 = cornerNormalAt(c1);
                    if (!isFiniteVec(n0) || !isFiniteVec(n1)) continue;

                    const float diff = corner[c0].value - corner[c1].value;
                    float t = 0.0f;
                    if (std::isfinite(diff) && std::abs(diff) >= kInterpEpsilon)
                        t = std::max(0.0f, std::min(1.0f, corner[c0].value / diff));

                    // A cancelled endpoint pair blends to the zero vector; refuse the
                    // edge instead of normalizing it into a NaN or a fabricated normal.
                    const Eigen::Vector3f blended = n0 + t * (n1 - n0);
                    const float blend_len = blended.norm();
                    if (!std::isfinite(blend_len) || blend_len <= kGradientEpsilon) continue;

                    edge_verts[e]  = vtx;
                    edge_norms[e]  = blended / blend_len;
                    edge_colors[e][0] = static_cast<uint8_t>(corner_vox[c0].r + t * (static_cast<float>(corner_vox[c1].r) - corner_vox[c0].r));
                    edge_colors[e][1] = static_cast<uint8_t>(corner_vox[c0].g + t * (static_cast<float>(corner_vox[c1].g) - corner_vox[c0].g));
                    edge_colors[e][2] = static_cast<uint8_t>(corner_vox[c0].b + t * (static_cast<float>(corner_vox[c1].b) - corner_vox[c0].b));
                    edge_ok[e] = true;
                }

                for (int t = 0; tables::tri_table[cube_idx][t] != -1; t += 3) {
                    const int ea = tables::tri_table[cube_idx][t + 0];
                    const int eb = tables::tri_table[cube_idx][t + 1];
                    const int ec = tables::tri_table[cube_idx][t + 2];
                    // A triangle needs all three of its vertices; refusing one edge
                    // costs only the rows that reference it. An edge the mask never
                    // marked is never read, so a table slip cannot emit garbage here.
                    if (!edge_ok[ea] || !edge_ok[eb] || !edge_ok[ec]) continue;
                    const int tri_edges[3] = {ea, eb, ec};
                    // Reverse winding (2, 1, 0 instead of 0, 1, 2) to fix front-face culling
                    for (int i = 2; i >= 0; --i) {
                        int e = tri_edges[i];
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
