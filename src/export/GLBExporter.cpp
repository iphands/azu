#include "export/GLBExporter.h"
#include "export/ColorConversion.h"
#include "utils/ColorMath.h"
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <vector>
#include <unordered_map>
#include "utils/Logger.h"

// Forward declarations/Includes for tinygltf
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_JSON
#include "json.hpp"
#include "tiny_gltf.h"

namespace kfusion {
namespace export_io {

namespace {

// Coordinate mapping (GLB-09). Both spaces are RIGHT-HANDED; glTF's canonical
// orientation is X right, Y UP, Z backward (towards the viewer), while the project's
// source space is X right, Y DOWN, Z forward. Flipping the two axes that differ
// gives (x, y, z) -> (x, -y, -z), a rotation (determinant +1), so handedness AND
// triangle winding are both preserved and no node transform is needed.
static Eigen::Vector3f toGLTF(const Eigen::Vector3f& v) {
    return Eigen::Vector3f(v.x(), -v.y(), -v.z());
}

// glTF requires NORMAL accessor vectors to be unit length. A zero-length (or
// underflow-to-zero) normal therefore cannot be written as-is, and silently
// dropping the whole NORMAL attribute would change the file schema for one bad
// vertex. The documented contract is a deterministic substitute instead: a normal
// that cannot be normalized becomes kFallbackNormalSource - source-space +Z, the
// camera-forward direction - taken through the very same mapping as every other
// normal, i.e. (0, 0, -1) in glTF space. It is unit length, it is constant, and it
// is reported per export, so the output is reproducible and never the (0,0,0) that
// a strict reader rejects (GLB-05).
constexpr float kMinNormalLength = 1e-6f;
const Eigen::Vector3f kFallbackNormalSource(0.0f, 0.0f, 1.0f);

// Length is measured before the mapping: toGLTF only flips signs, so it preserves
// length, and measuring here makes the substitution decision depend on the input
// alone. `substituted` is set exactly when the fallback was written.
Eigen::Vector3f unitNormalToGLTF(const Eigen::Vector3f& n_source, bool& substituted) {
    substituted = false;
    const float len = n_source.norm();
    if (!std::isfinite(len) || len < kMinNormalLength) {
        substituted = true;
        return toGLTF(kFallbackNormalSource);
    }
    return toGLTF(n_source) / len;
}

// The target must be openable before the writer runs, because tinygltf creates the
// file the instant it opens it: after the fact, an unopenable path and a truncated
// write look identical except for what the filesystem already said. Existence is
// the only thing checked here (never permissions, which a privileged caller would
// misreport), so nothing below can refuse a path the writer could have opened.
bool targetPrecheck(const std::string& filepath, std::string& reason) {
    std::error_code ec;
    const std::filesystem::path path(filepath);
    if (std::filesystem::is_directory(path, ec)) {
        reason = "target path is an existing directory";
        return false;
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty() && !std::filesystem::is_directory(parent, ec)) {
        reason = "parent directory " + parent.string() + " does not exist";
        return false;
    }
    return true;
}

// GLB-06: tinygltf's writer takes no error out-parameter, so nothing it hands back
// can name a cause. What is observable is what the write path left behind: the errno
// of the failed I/O and the state of the target. That is what gets reported.
std::string diagnoseWriteFailure(const std::string& filepath, int writer_errno) {
    std::string s = "tinygltf WriteGltfSceneToFile returned false";
    if (writer_errno != 0) {
        s += "; errno=" + std::to_string(writer_errno) + " (" + std::strerror(writer_errno) + ")";
    } else {
        s += "; no errno was set by the write path";
    }
    std::error_code ec;
    if (std::filesystem::is_regular_file(filepath, ec)) {
        s += "; target left a " + std::to_string(std::filesystem::file_size(filepath, ec)) +
             " byte(s) partial file";
    } else if (std::filesystem::exists(filepath, ec)) {
        s += "; target exists but is not a regular file";
    } else {
        s += "; no file was created";
    }
    return s;
}

// Delete an unfinished GLB, but only when the target IS a regular file: a failed
// write may name a directory, and destroying a target this writer never created is a
// worse bug than the partial file it would have deleted.
bool removePartialFile(const std::string& filepath) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(filepath, ec)) return false;
    return std::filesystem::remove(filepath, ec) && !ec;
}

} // namespace

bool GLBExporter::write(const meshing::MeshData& input_mesh, const std::string& filepath) {
    // Runs before the weld loop, which indexes positions/normals/colors with
    // these raw indices: an out-of-range index is UB before any file is opened.
    std::string reason;
    if (!input_mesh.validate(&reason)) {
        KFLOGF_ERROR("GLBExport", "Invalid mesh, nothing written to %s: %s",
                     filepath.c_str(), reason.c_str());
        return false;
    }

    // Same fail-before-open discipline as validate(): a path that cannot hold the
    // file is refused here, with the cause, before tinygltf can create anything.
    if (!targetPrecheck(filepath, reason)) {
        KFLOGF_ERROR("GLBExport", "Refusing to write %s: %s", filepath.c_str(), reason.c_str());
        return false;
    }

    KFLOGF_INFO("GLBExport", "Starting export to %s", filepath.c_str());
    KFLOGF_INFO("GLBExport", "Original mesh: %zu vertices, %zu indices", input_mesh.positions.size(), input_mesh.indices.size());

    meshing::MeshData mesh;
    // WELD VERTICES
    struct VertexHash {
        size_t operator()(const Eigen::Vector3f& v) const {
            size_t h1 = std::hash<float>{}(v.x());
            size_t h2 = std::hash<float>{}(v.y());
            size_t h3 = std::hash<float>{}(v.z());
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
    std::unordered_map<Eigen::Vector3f, uint32_t, VertexHash> vertex_map;
    bool in_has_normals = (input_mesh.normals.size() == input_mesh.positions.size());
    bool in_has_colors = (input_mesh.colors.size() == input_mesh.positions.size() * 3);
    
    mesh.indices.reserve(input_mesh.indices.size());
    
    for (size_t i = 0; i < input_mesh.indices.size(); ++i) {
        uint32_t orig_idx = input_mesh.indices[i];
        const Eigen::Vector3f& pos = input_mesh.positions[orig_idx];
        
        auto it = vertex_map.find(pos);
        if (it != vertex_map.end()) {
            mesh.indices.push_back(it->second);
        } else {
            uint32_t new_idx = static_cast<uint32_t>(mesh.positions.size());
            vertex_map[pos] = new_idx;
            mesh.positions.push_back(pos);
            if (in_has_normals) mesh.normals.push_back(input_mesh.normals[orig_idx]);
            if (in_has_colors) {
                mesh.colors.push_back(input_mesh.colors[orig_idx*3+0]);
                mesh.colors.push_back(input_mesh.colors[orig_idx*3+1]);
                mesh.colors.push_back(input_mesh.colors[orig_idx*3+2]);
            }
            mesh.indices.push_back(new_idx);
        }
    }

    const size_t nvert = mesh.positions.size();
    const size_t nidx  = mesh.indices.size();
    const bool has_normals = (mesh.normals.size() == nvert);
    const bool has_colors  = (mesh.colors.size() == nvert * 3);

    KFLOGF_INFO("GLBExport", "Welded mesh: %zu vertices, %zu indices (has_normals=%d, has_colors=%d)", 
                nvert, nidx, has_normals, has_colors);

    // Defense-in-depth before any accessor/binary payload is derived from the
    // weld, and semantics-preserving: the weld only copies already-validated
    // positions/normals/colors and rebuilds indices from new_idx, so a valid
    // input always welds to a valid mesh. Firing means the weld itself broke.
    if (!mesh.validate(&reason)) {
        KFLOGF_ERROR("GLBExport", "Welded mesh is invalid, nothing written to %s: %s",
                     filepath.c_str(), reason.c_str());
        return false;
    }

    // ---------------------------------------------------------
    // Build binary buffer
    // ---------------------------------------------------------

    // Layout: [positions | normals | colors | indices]
    std::vector<uint8_t> buffer_data;

    auto appendBytes = [&](const void* data, size_t bytes) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        buffer_data.insert(buffer_data.end(), p, p + bytes);
        // Pad to 4-byte alignment
        while (buffer_data.size() % 4 != 0) buffer_data.push_back(0);
    };

    // --- positions ---
    size_t pos_offset = buffer_data.size();
    {
        std::vector<float> pos_data;
        pos_data.reserve(nvert * 3);
        for (size_t i = 0; i < nvert; ++i) {
            Eigen::Vector3f p = toGLTF(mesh.positions[i]);
            pos_data.push_back(p.x());
            pos_data.push_back(p.y());
            pos_data.push_back(p.z());
        }
        appendBytes(pos_data.data(), pos_data.size() * sizeof(float));
    }
    size_t pos_length = buffer_data.size() - pos_offset;

    // --- normals ---
    size_t norm_offset = 0;
    size_t norm_length = 0;
    size_t fallback_normals = 0;
    if (has_normals) {
        norm_offset = buffer_data.size();
        std::vector<float> norm_data;
        norm_data.reserve(nvert * 3);
        for (size_t i = 0; i < nvert; ++i) {
            bool substituted = false;
            const Eigen::Vector3f n = unitNormalToGLTF(mesh.normals[i], substituted);
            if (substituted) ++fallback_normals;
            norm_data.push_back(n.x());
            norm_data.push_back(n.y());
            norm_data.push_back(n.z());
        }
        appendBytes(norm_data.data(), norm_data.size() * sizeof(float));
        norm_length = buffer_data.size() - norm_offset;
    }
    if (fallback_normals > 0) {
        KFLOGF_INFO("GLBExport", "%zu normal(s) could not be normalized; the documented "
                                 "fallback (source +Z -> glTF (0,0,-1)) was written in their place",
                    fallback_normals);
    }

    // --- vertex colors (as vec4 FLOAT, LINEAR light) ---
    // MeshData colors are uint8 sRGB; glTF COLOR_0 is linear float, so this is
    // the mandatory decode (docs/CANONICAL_SEMANTICS.md). Alpha is not converted.
    // The checked decode rejects rather than clamps, and it runs while the mesh
    // is still in memory: a rejected component aborts the export BEFORE the file
    // is opened, exactly like the validate() gates above.
    size_t col_offset = 0;
    size_t col_length = 0;
    std::vector<float> col_data;
    if (has_colors) {
        col_data.reserve(nvert * 4);
        for (size_t i = 0; i < nvert; ++i) {
            float lr = 0.0f, lg = 0.0f, lb = 0.0f;
            if (!srgbColorToLinear(utils::srgbUint8ToFloat(mesh.colors[i*3+0]),
                                   utils::srgbUint8ToFloat(mesh.colors[i*3+1]),
                                   utils::srgbUint8ToFloat(mesh.colors[i*3+2]), lr, lg, lb)) {
                KFLOGF_ERROR("GLBExport", "Color component %zu is not a valid sRGB value, "
                                          "nothing written to %s", i, filepath.c_str());
                return false;
            }
            col_data.push_back(lr);
            col_data.push_back(lg);
            col_data.push_back(lb);
            col_data.push_back(1.0f); // alpha
        }
        col_offset = buffer_data.size();
        appendBytes(col_data.data(), col_data.size() * sizeof(float));
        col_length = buffer_data.size() - col_offset;
    }

    // --- indices ---
    size_t idx_offset = buffer_data.size();
    appendBytes(mesh.indices.data(), nidx * sizeof(uint32_t));
    size_t idx_length = buffer_data.size() - idx_offset;

    KFLOGF_DEBUG("GLBExport", "Buffer views constructed. Total binary payload: %zu bytes", buffer_data.size());
    KFLOGF_DEBUG("GLBExport", "  Position: offset=%zu, length=%zu", pos_offset, pos_length);
    if (has_normals) KFLOGF_DEBUG("GLBExport", "  Normal: offset=%zu, length=%zu", norm_offset, norm_length);
    if (has_colors)  KFLOGF_DEBUG("GLBExport", "  Color: offset=%zu, length=%zu", col_offset, col_length);
    KFLOGF_DEBUG("GLBExport", "  Indices: offset=%zu, length=%zu", idx_offset, idx_length);

    // ---------------------------------------------------------
    // Build GLTF model
    // ---------------------------------------------------------
    tinygltf::Model model;
    model.asset.version   = "2.0";
    model.asset.generator = "KinectFusionQt";

    // Buffer
    tinygltf::Buffer gltf_buf;
    // GLB-08: the payload is moved into the writer's buffer representation. buffer_data
    // has no reader after this point (every offset and length above was captured while
    // it was still intact), so copying the largest object in the export would be pure
    // waste.
    gltf_buf.data = std::move(buffer_data);
    model.buffers.push_back(std::move(gltf_buf));

    int buf_idx = 0;

    // Helper: add buffer view
    auto addBufferView = [&](size_t offset, size_t length, int target) -> int {
        tinygltf::BufferView bv;
        bv.buffer     = buf_idx;
        bv.byteOffset = offset;
        bv.byteLength = length;
        bv.target     = target;
        model.bufferViews.push_back(bv);
        return static_cast<int>(model.bufferViews.size()) - 1;
    };

    // Helper: add accessor
    auto addAccessor = [&](int bv_idx, int component_type, int type, size_t count,
                           bool normalized = false,
                           bool has_minmax = false,
                           std::vector<double> min_v = {},
                           std::vector<double> max_v = {}) -> int {
        tinygltf::Accessor acc;
        acc.bufferView    = bv_idx;
        acc.byteOffset    = 0;
        acc.componentType = component_type;
        acc.type          = type;
        acc.count         = count;
        acc.normalized    = normalized;
        if (has_minmax) { acc.minValues = min_v; acc.maxValues = max_v; }
        model.accessors.push_back(acc);
        return static_cast<int>(model.accessors.size()) - 1;
    };

    // Position buffer view + accessor
    int bv_pos = addBufferView(pos_offset, pos_length, TINYGLTF_TARGET_ARRAY_BUFFER);

    // Compute bounding box for accessor min/max (required for positions). Seeded from
    // the first vertex, not from a sentinel: a +-1e9 seed silently clips an axis whose
    // coordinates all lie beyond it, and these values are a promise to the reader
    // about the geometry it just loaded.
    Eigen::Vector3f bmin = toGLTF(mesh.positions[0]);
    Eigen::Vector3f bmax = bmin;
    for (size_t i = 1; i < nvert; ++i) {
        const Eigen::Vector3f pg = toGLTF(mesh.positions[i]);
        bmin = bmin.cwiseMin(pg);
        bmax = bmax.cwiseMax(pg);
    }
    int acc_pos = addAccessor(bv_pos, TINYGLTF_COMPONENT_TYPE_FLOAT,
                              TINYGLTF_TYPE_VEC3, nvert, false, true,
                              {bmin.x(), bmin.y(), bmin.z()},
                              {bmax.x(), bmax.y(), bmax.z()});

    int acc_norm = -1;
    if (has_normals) {
        int bv_norm = addBufferView(norm_offset, norm_length, TINYGLTF_TARGET_ARRAY_BUFFER);
        acc_norm = addAccessor(bv_norm, TINYGLTF_COMPONENT_TYPE_FLOAT,
                               TINYGLTF_TYPE_VEC3, nvert);
    }

    int acc_col = -1;
    if (has_colors) {
        int bv_col = addBufferView(col_offset, col_length, TINYGLTF_TARGET_ARRAY_BUFFER);
        acc_col = addAccessor(bv_col, TINYGLTF_COMPONENT_TYPE_FLOAT,
                              TINYGLTF_TYPE_VEC4, nvert);
    }

    int bv_idx_gltf = addBufferView(idx_offset, idx_length, TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER);
    int acc_idx = addAccessor(bv_idx_gltf, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,
                              TINYGLTF_TYPE_SCALAR, nidx);

    // Mesh primitive
    tinygltf::Primitive primitive;
    primitive.mode               = TINYGLTF_MODE_TRIANGLES;
    primitive.attributes["POSITION"] = acc_pos;
    if (acc_norm >= 0) primitive.attributes["NORMAL"] = acc_norm;
    if (acc_col  >= 0) primitive.attributes["COLOR_0"] = acc_col;
    primitive.indices            = acc_idx;
    primitive.material           = 0;

    tinygltf::Mesh gltf_mesh;
    gltf_mesh.name = "KinectScan";
    gltf_mesh.primitives.push_back(primitive);
    model.meshes.push_back(gltf_mesh);

    // Default material
    tinygltf::Material mat;
    mat.name = "ScanMaterial";
    if (has_colors) {
        mat.pbrMetallicRoughness.baseColorFactor = {1.0, 1.0, 1.0, 1.0};
    } else {
        mat.pbrMetallicRoughness.baseColorFactor = {0.72, 0.78, 0.85, 1.0};
    }
    mat.pbrMetallicRoughness.metallicFactor  = 0.0;
    mat.pbrMetallicRoughness.roughnessFactor = 0.8;
    model.materials.push_back(mat);

    // Scene node
    tinygltf::Node node;
    node.mesh = 0;
    // No extra transform: 1 unit = 1 meter, Y-up, already handled by coordinate flip
    model.nodes.push_back(node);

    tinygltf::Scene scene;
    scene.nodes.push_back(0);
    model.scenes.push_back(scene);
    model.defaultScene = 0;

    // ---------------------------------------------------------
    // Write GLB
    // ---------------------------------------------------------
    tinygltf::TinyGLTF writer;
    const bool ok = writer.WriteGltfSceneToFile(&model, filepath,
        /*embedImages=*/true,
        /*embedBuffers=*/true,
        /*prettyPrint=*/false,
        /*writeBinary=*/true);

    if (!ok) {
        // Captured before any further filesystem call can overwrite it.
        const int writer_errno = errno;
        const std::string cause = diagnoseWriteFailure(filepath, writer_errno);
        const bool cleaned = removePartialFile(filepath);
        KFLOGF_ERROR("GLBExport", "GLB write failed for %s: %s%s", filepath.c_str(), cause.c_str(),
                     cleaned ? "; partial file removed" :
                               "; nothing to remove (the target is not a file we created)");
        return false;
    }

    KFLOGF_INFO("GLBExport", "Exported %zu verts, %zu tris to: %s", nvert, nidx/3, filepath.c_str());
    return true;
}

} // namespace export_io
} // namespace kfusion
