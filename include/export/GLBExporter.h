#pragma once

#include "meshing/MeshData.h"
#include <string>

namespace kfusion {
namespace export_io {

class GLBExporter {
public:
    // Write a glTF 2.0 binary (.glb) scene: right-handed, Y-up, Z toward the viewer,
    // 1 unit = 1 metre, no node transform. Vertex colors are written as COLOR_0 in
    // linear float RGB; a mesh that fails MeshData::validate(), or a target that
    // cannot hold the file, produces no file at all.
    // Returns true on success.
    static bool write(const meshing::MeshData& mesh,
                      const std::string& filepath);
};

} // namespace export_io
} // namespace kfusion
