#include "export/PLYExporter.h"
#include <fstream>
#include <iostream>
#include <cstring>

namespace kfusion {
namespace export_io {

bool PLYExporter::writeBinary(const meshing::MeshData& mesh,
                               const std::string& filepath)
{
    std::string reason;
    if (!mesh.validate(&reason)) {
        std::cerr << "[PLY] Invalid mesh, nothing written to " << filepath << ": "
                  << reason << "\n";
        return false;
    }

    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[PLY] Cannot open: " << filepath << "\n";
        return false;
    }

    const size_t nvert = mesh.positions.size();
    const size_t nface = mesh.indices.size() / 3;
    const bool has_color = (mesh.colors.size() == nvert * 3);
    const bool has_normals = (mesh.normals.size() == nvert);

    // Write ASCII header
    f << "ply\n";
    f << "format binary_little_endian 1.0\n";
    f << "comment KinectFusionQt export\n";
    f << "element vertex " << nvert << "\n";
    f << "property float x\n";
    f << "property float y\n";
    f << "property float z\n";
    if (has_normals) {
        f << "property float nx\n";
        f << "property float ny\n";
        f << "property float nz\n";
    }
    if (has_color) {
        f << "property uint8 red\n";
        f << "property uint8 green\n";
        f << "property uint8 blue\n";
    }
    f << "element face " << nface << "\n";
    f << "property list uint8 int32 vertex_indices\n";
    f << "end_header\n";

    // Write vertex data
    for (size_t i = 0; i < nvert; ++i) {
        float x = mesh.positions[i].x();
        float y = mesh.positions[i].y();
        float z = mesh.positions[i].z();
        f.write(reinterpret_cast<const char*>(&x), 4);
        f.write(reinterpret_cast<const char*>(&y), 4);
        f.write(reinterpret_cast<const char*>(&z), 4);

        if (has_normals) {
            float nx = mesh.normals[i].x();
            float ny = mesh.normals[i].y();
            float nz = mesh.normals[i].z();
            f.write(reinterpret_cast<const char*>(&nx), 4);
            f.write(reinterpret_cast<const char*>(&ny), 4);
            f.write(reinterpret_cast<const char*>(&nz), 4);
        }

        if (has_color) {
            f.write(reinterpret_cast<const char*>(&mesh.colors[i*3+0]), 1);
            f.write(reinterpret_cast<const char*>(&mesh.colors[i*3+1]), 1);
            f.write(reinterpret_cast<const char*>(&mesh.colors[i*3+2]), 1);
        }
    }

    // Write face data
    for (size_t t = 0; t < nface; ++t) {
        uint8_t count = 3;
        f.write(reinterpret_cast<const char*>(&count), 1);
        for (int k = 0; k < 3; ++k) {
            uint32_t idx = mesh.indices[t * 3 + k];
            f.write(reinterpret_cast<const char*>(&idx), 4);
        }
    }

    if (!f.good()) {
        std::cerr << "[PLY] Write error on: " << filepath << "\n";
        return false;
    }

    std::cout << "[PLY] Exported " << nvert << " vertices, " << nface
              << " triangles to: " << filepath << "\n";
    return true;
}

bool PLYExporter::writeASCII(const meshing::MeshData& mesh,
                              const std::string& filepath)
{
    std::string reason;
    if (!mesh.validate(&reason)) {
        std::cerr << "[PLY] Invalid mesh, nothing written to " << filepath << ": "
                  << reason << "\n";
        return false;
    }

    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "[PLY] Cannot open: " << filepath << "\n";
        return false;
    }

    const size_t nvert = mesh.positions.size();
    const size_t nface = mesh.indices.size() / 3;
    const bool has_color = (mesh.colors.size() == nvert * 3);
    const bool has_normals = (mesh.normals.size() == nvert);

    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << nvert << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    if (has_normals) f << "property float nx\nproperty float ny\nproperty float nz\n";
    if (has_color)   f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "element face " << nface << "\n";
    f << "property list uchar uint vertex_indices\nend_header\n";

    for (size_t i = 0; i < nvert; ++i) {
        f << mesh.positions[i].x() << " " << mesh.positions[i].y() << " " << mesh.positions[i].z();
        if (has_normals)
            f << " " << mesh.normals[i].x() << " " << mesh.normals[i].y() << " " << mesh.normals[i].z();
        if (has_color)
            f << " " << static_cast<int>(mesh.colors[i*3+0])
              << " " << static_cast<int>(mesh.colors[i*3+1])
              << " " << static_cast<int>(mesh.colors[i*3+2]);
        f << "\n";
    }

    for (size_t t = 0; t < nface; ++t) {
        f << "3 " << mesh.indices[t*3] << " " << mesh.indices[t*3+1] << " " << mesh.indices[t*3+2] << "\n";
    }

    std::cout << "[PLY] ASCII exported: " << filepath << "\n";
    return f.good();
}

} // namespace export_io
} // namespace kfusion
