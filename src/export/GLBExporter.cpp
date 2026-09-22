#include "export/GLBExporter.h"
#include "export/ColorConversion.h"
#include "utils/ColorMath.h"
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

// Kinect → Unity/GLTF coordinate conversion:
//   Kinect: right-handed, X right, Y down, Z forward
//   GLTF:   right-handed, X right, Y up,   Z backward
// Mapping:
//   (x, y, z) -> (x, -y, -z)
// Flipping two axes (Y and Z) preserves winding order.
static Eigen::Vector3f toGLTF(const Eigen::Vector3f& v) {
    return Eigen::Vector3f(v.x(), -v.y(), -v.z());
}

bool GLBExporter::write(const meshing::MeshData& input_mesh, const std::string& filepath) {
    // Runs before the weld loop, which indexes positions/normals/colors with
    // these raw indices: an out-of-range index is UB before any file is opened.
    std::string reason;
    if (!input_mesh.validate(&reason)) {
        KFLOGF_ERROR("GLBExport", "Invalid mesh, nothing written to %s: %s",
                     filepath.c_str(), reason.c_str());
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
    if (has_normals) {
        norm_offset = buffer_data.size();
        std::vector<float> norm_data;
        norm_data.reserve(nvert * 3);
        for (size_t i = 0; i < nvert; ++i) {
            Eigen::Vector3f n = toGLTF(mesh.normals[i]);
            // Re-normalize after transformation
            float len = n.norm();
            if (len > 1e-6f) n /= len;
            norm_data.push_back(n.x());
            norm_data.push_back(n.y());
            norm_data.push_back(n.z());
        }
        appendBytes(norm_data.data(), norm_data.size() * sizeof(float));
        norm_length = buffer_data.size() - norm_offset;
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
    gltf_buf.data = buffer_data;
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

    // Compute bounding box for accessor min/max (required for positions)
    Eigen::Vector3f bmin( 1e9f,  1e9f,  1e9f);
    Eigen::Vector3f bmax(-1e9f, -1e9f, -1e9f);
    for (const auto& p : mesh.positions) {
        Eigen::Vector3f pg = toGLTF(p);
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
    std::string err, warn;
    bool ok = writer.WriteGltfSceneToFile(&model, filepath,
        /*embedImages=*/true,
        /*embedBuffers=*/true,
        /*prettyPrint=*/false,
        /*writeBinary=*/true);

    if (!ok) {
        KFLOGF_ERROR("GLBExport", "WriteGltfSceneToFile failed for: %s", filepath.c_str());
        if (!err.empty())  KFLOGF_ERROR("GLBExport", "  Error: %s", err.c_str());
        if (!warn.empty()) KFLOGF_WARN("GLBExport", "  Warn:  %s", warn.c_str());
        return false;
    }

    KFLOGF_INFO("GLBExport", "Exported %zu verts, %zu tris to: %s", nvert, nidx/3, filepath.c_str());
    return true;
}

} // namespace export_io
} // namespace kfusion
