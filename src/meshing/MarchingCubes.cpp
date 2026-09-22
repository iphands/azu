#include "meshing/MarchingCubes.h"
#include "meshing/MarchingCubesTables.h"
#include "utils/ColorMath.h"
#include <cmath>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

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

// Canonical identity of a physical crossing edge: the LOWER endpoint voxel plus the
// single axis the two endpoints differ on. Two cubes (or a cube across an OpenMP
// slice boundary) that share a physical edge derive the identical key, so the
// global weld merges them exactly once. This replaces the pre-Todo-17 spatial
// quantization, which merged vertices only when their interpolated positions
// rounded into the same bucket - falsely splitting FP-variant shared edges and
// falsely merging distinct edges that happened to collide (meshing:D8).
struct EdgeKey {
    int     x, y, z;
    uint8_t axis;
    bool operator==(const EdgeKey& o) const {
        return x == o.x && y == o.y && z == o.z && axis == o.axis;
    }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
        // mix distinct primes over the four small fields
        size_t h = static_cast<size_t>(k.x) * 0x9E3779B97F4A7C15ull;
        h ^= (static_cast<size_t>(static_cast<uint32_t>(k.y)) + 0x9E3779B9u + (h << 6) + (h >> 2));
        h ^= (static_cast<size_t>(static_cast<uint32_t>(k.z)) + 0x9E3779B9u + (h << 6) + (h >> 2));
        h ^= (static_cast<size_t>(k.axis) + 0x9E3779B9u + (h << 6) + (h >> 2));
        return h;
    }
};

} // namespace

MarchingCubes::EdgeCrossing MarchingCubes::crossingFromLower(
    const Eigen::Vector3f& p_low, float v_low,
    const Eigen::Vector3f& p_up,  float v_up)
{
    if (!std::isfinite(v_low) || !std::isfinite(v_up) ||
        !isFiniteVec(p_low) || !isFiniteVec(p_up))
        return {invalidNormal(), 0.0f, false};
    // A zero endpoint IS the crossing, so return that endpoint exactly. `t` is
    // measured from the lower endpoint (0 at low, 1 at up), so every cube that
    // reaches this physical edge from any orientation obtains the identical `t`,
    // position, normal and color - the shared-parameter guarantee that the old
    // position-only interpolateEdge could not carry (meshing:D8).
    if (std::abs(v_low) < kInterpEpsilon) return {p_low, 0.0f, true};
    if (std::abs(v_up)  < kInterpEpsilon) return {p_up,  1.0f, true};
    const float diff = v_low - v_up;
    if (!std::isfinite(diff) || std::abs(diff) < kInterpEpsilon)
        return {p_low, 0.0f, true};   // collapsed blend: deterministic lower endpoint
    const float t = std::max(0.0f, std::min(1.0f, v_low / diff));
    return {p_low + t * (p_up - p_low), t, true};
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
    const auto& p = volume.params();
    const int   RES = p.resolution;

    auto out = std::make_shared<MeshData>();

    // Fewer than two voxels per axis contains no cube; return an empty mesh rather
    // than sizing slice arrays to (RES - 1) == 0 or dividing a progress span by
    // (RES - 2) == 0. The final progress call is still emitted, exactly 1.0.
    if (RES < 2) {
        if (progress_cb) progress_cb(1.0f);
        return out;
    }

    // Per-slice local output. Every emitted local vertex carries the canonical
    // EdgeKey of the physical edge it came from, so the serial merge can weld by
    // exact edge identity instead of a quantized world position.
    struct SliceOut {
        std::vector<Eigen::Vector3f> pos;
        std::vector<Eigen::Vector3f> nrm;
        std::vector<uint8_t>         col;
        std::vector<EdgeKey>         key;
        std::vector<uint32_t>        idx;
    };
    std::vector<SliceOut> slices(RES - 1);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int z = 0; z < RES - 1; ++z) {
        auto& sl = slices[z];

        for (int y = 0; y < RES - 1; ++y) {
            for (int x = 0; x < RES - 1; ++x) {
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

                Eigen::Vector3f edge_pos[12];
                Eigen::Vector3f edge_norms[12];
                uint8_t         edge_colors[12][3];
                EdgeKey         edge_key[12];
                bool            edge_ok[12] = {false, false, false, false,
                                               false, false, false, false,
                                               false, false, false, false};

                for (int e = 0; e < 12; ++e) {
                    if (!(edge_mask & (1 << e))) continue;
                    const int c0 = EDGE_CORNERS[e][0];
                    const int c1 = EDGE_CORNERS[e][1];
                    if (!corner[c0].supported || !corner[c1].supported) continue;

                    // Canonical edge identity + which endpoint is the LOWER one. A unit
                    // edge varies on exactly one axis; the lower endpoint is the corner
                    // with the smaller coordinate on that axis. Keying and orienting by
                    // the lower endpoint makes the payload identical for every cube that
                    // reaches this physical edge, from any direction or slice.
                    const int d0x = x + CORNER_OFFSETS[c0][0], d0y = y + CORNER_OFFSETS[c0][1], d0z = z + CORNER_OFFSETS[c0][2];
                    const int d1x = x + CORNER_OFFSETS[c1][0], d1y = y + CORNER_OFFSETS[c1][1], d1z = z + CORNER_OFFSETS[c1][2];
                    int axis, low_c, up_c;
                    if (d0x != d1x)      { axis = 0; low_c = (d0x <= d1x) ? c0 : c1; up_c = (d0x <= d1x) ? c1 : c0; }
                    else if (d0y != d1y) { axis = 1; low_c = (d0y <= d1y) ? c0 : c1; up_c = (d0y <= d1y) ? c1 : c0; }
                    else                 { axis = 2; low_c = (d0z <= d1z) ? c0 : c1; up_c = (d0z <= d1z) ? c1 : c0; }
                    edge_key[e] = EdgeKey{ x + CORNER_OFFSETS[low_c][0],
                                           y + CORNER_OFFSETS[low_c][1],
                                           z + CORNER_OFFSETS[low_c][2],
                                           static_cast<uint8_t>(axis) };

                    // ONE interpolation parameter drives position, normal and color.
                    const EdgeCrossing cr = crossingFromLower(
                        corner_pos[low_c], corner[low_c].value,
                        corner_pos[up_c],  corner[up_c].value);
                    if (!cr.ok) continue;

                    const Eigen::Vector3f n_low = cornerNormalAt(low_c);
                    const Eigen::Vector3f n_up  = cornerNormalAt(up_c);
                    if (!isFiniteVec(n_low) || !isFiniteVec(n_up)) continue;

                    // A cancelled endpoint pair blends to the zero vector; refuse the
                    // edge instead of normalizing it into a NaN or a fabricated normal.
                    const Eigen::Vector3f blended = n_low + cr.t * (n_up - n_low);
                    const float blend_len = blended.norm();
                    if (!std::isfinite(blend_len) || blend_len <= kGradientEpsilon) continue;

                    edge_pos[e]   = cr.position;
                    edge_norms[e] = blended / blend_len;
                    // Endpoints are float sRGB, blended with the SAME t that produced
                    // the position (Todo 17's shared parameter) and quantized through
                    // the one CPU policy, which saturates a finite overshoot of the
                    // blend to the endpoint byte. Only a non-finite endpoint or blend
                    // invalidates this crossing edge outright: every triangle row
                    // touching it is dropped below, exactly like an unusable normal.
                    float col_f[3];
                    col_f[0] = corner_vox[low_c].r + cr.t * (corner_vox[up_c].r - corner_vox[low_c].r);
                    col_f[1] = corner_vox[low_c].g + cr.t * (corner_vox[up_c].g - corner_vox[low_c].g);
                    col_f[2] = corner_vox[low_c].b + cr.t * (corner_vox[up_c].b - corner_vox[low_c].b);
                    uint8_t col_b[3] = {0, 0, 0};
                    if (!utils::srgbFloatToUint8(col_f[0], col_b[0]) ||
                        !utils::srgbFloatToUint8(col_f[1], col_b[1]) ||
                        !utils::srgbFloatToUint8(col_f[2], col_b[2]))
                        continue;
                    edge_colors[e][0] = col_b[0];
                    edge_colors[e][1] = col_b[1];
                    edge_colors[e][2] = col_b[2];
                    edge_ok[e] = true;
                }

                for (int t = 0; tables::tri_table[cube_idx][t] != -1; t += 3) {
                    const int ea = tables::tri_table[cube_idx][t + 0];
                    const int eb = tables::tri_table[cube_idx][t + 1];
                    const int ec = tables::tri_table[cube_idx][t + 2];
                    if (!edge_ok[ea] || !edge_ok[eb] || !edge_ok[ec]) continue;
                    const int tri_edges[3] = {ea, eb, ec};
                    // Derived outward convention: reverse table order (2, 1, 0); forward (0, 1, 2)
                    // is inward under the signed-distance fixture. See docs/CANONICAL_SEMANTICS.md.
                    for (int i = 2; i >= 0; --i) {
                        const int e = tri_edges[i];
                        const uint32_t vidx = static_cast<uint32_t>(sl.pos.size());
                        sl.pos.push_back(edge_pos[e]);
                        sl.nrm.push_back(edge_norms[e]);
                        sl.col.push_back(edge_colors[e][0]);
                        sl.col.push_back(edge_colors[e][1]);
                        sl.col.push_back(edge_colors[e][2]);
                        sl.key.push_back(edge_key[e]);
                        sl.idx.push_back(vidx);
                    }
                }
            }
        }
    }

    // Serial, deterministic merge: slices in ascending z, triangles in local order,
    // welded by exact EdgeKey (first insertion wins). Because every cube derives the
    // same payload for a given key, first-wins is byte-identical regardless of which
    // cube or thread reached the edge first. The progress callback fires ONLY here,
    // from the calling thread, so it can never overlap and its sequence is a pure
    // function of the resolution. The triangle budget stops at a full-triangle
    // boundary and never emits a partial row.
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> weld;
    const size_t cap = max_triangles_cpu_;
    size_t tri_emitted = 0;
    bool   truncated   = false;

    for (int z = 0; z < RES - 1; ++z) {
        const SliceOut& sl = slices[z];
        for (size_t i = 0; i + 2 < sl.idx.size(); i += 3) {
            if (tri_emitted >= cap) { truncated = true; break; }
            for (int k = 0; k < 3; ++k) {
                const uint32_t li   = sl.idx[i + k];
                const EdgeKey& key  = sl.key[li];
                const auto     it   = weld.find(key);
                uint32_t       gi;
                if (it != weld.end()) {
                    gi = it->second;
                } else {
                    gi = static_cast<uint32_t>(out->positions.size());
                    weld.emplace(key, gi);
                    out->positions.push_back(sl.pos[li]);
                    out->normals.push_back(sl.nrm[li]);
                    out->colors.push_back(sl.col[li * 3 + 0]);
                    out->colors.push_back(sl.col[li * 3 + 1]);
                    out->colors.push_back(sl.col[li * 3 + 2]);
                }
                out->indices.push_back(gi);
            }
            ++tri_emitted;
        }
        if (progress_cb)
            progress_cb(static_cast<float>(z + 1) / static_cast<float>(RES - 1));
    }

    out->truncated = truncated;
    if (progress_cb) progress_cb(1.0f);
    return out;
}

} // namespace meshing
} // namespace kfusion
