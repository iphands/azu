#pragma once

#include <QOpenGLFunctions_3_3_Core>
#include "rendering/ShaderProgram.h"
#include "rendering/Camera.h"
#include "sensor/FrameData.h"
#include "meshing/MeshData.h"
#include <memory>

namespace kfusion {
namespace rendering {

enum class RenderMode {
    PointCloud,
    Mesh
};

class PreviewRenderer : protected QOpenGLFunctions_3_3_Core {
public:
    PreviewRenderer();
    ~PreviewRenderer();

    void initialize();
    void resize(int w, int h);
    void render();

    void setMode(RenderMode mode) { mode_ = mode; }
    RenderMode mode() const { return mode_; }

    // Upload point cloud from frame data (standard raycasted model or live frame)
    void uploadPointCloud(const sensor::FrameData& frame);

    // Upload mesh for rendering
    void uploadMesh(const meshing::MeshData& mesh);

    /** Clear uploaded point cloud and mesh (e.g. after reset scan). */
    void clearGeometry();

    /** TSDF volume cage indicator: min corner and extent (state only, no GL). */
    void setVolumeBox(const Eigen::Vector3f& origin, const Eigen::Vector3f& size);
    void setVolumeBoxVisible(bool visible) { cage_visible_ = visible; }
    /** Amber tint when the tracked camera pose has left the volume. */
    void setVolumeBoxOutside(bool outside) { cage_outside_ = outside; }

    Camera& camera() { return camera_; }

private:
    RenderMode   mode_      = RenderMode::PointCloud;
    int          viewport_w_ = 1;
    int          viewport_h_ = 1;
    bool         initialized_ = false;

    Camera  camera_;

    // Volume cage (state valid pre-init; buffers built in initialize())
    Eigen::Vector3f cage_origin_{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f cage_size_{0.0f, 0.0f, 0.0f};
    bool cage_visible_ = true;
    bool cage_outside_ = false;
    float cage_line_width_ = 1.0f;
    unsigned int cage_vao_ = 0, cage_vbo_ = 0;
    ShaderProgram cage_shader_;

    // Point cloud GL objects
    unsigned int pc_vao_ = 0, pc_vbo_pos_ = 0, pc_vbo_col_ = 0;
    int          pc_count_ = 0;
    ShaderProgram pc_shader_;

    // Mesh GL objects
    unsigned int mesh_vao_ = 0, mesh_vbo_pos_ = 0, mesh_vbo_norm_ = 0, mesh_vbo_col_ = 0, mesh_ebo_ = 0;
    int          mesh_index_count_ = 0;
    ShaderProgram mesh_shader_;

    void initPointCloudBuffers();
    void initMeshBuffers();
    void initCageBuffers();
    void renderPointCloud();
    void renderMesh();
    void renderCage();

    // Shader sources
    static const char* POINTCLOUD_VERT;
    static const char* POINTCLOUD_FRAG;
    static const char* MESH_VERT;
    static const char* MESH_FRAG;
    static const char* CAGE_VERT;
    static const char* CAGE_FRAG;
};

} // namespace rendering
} // namespace kfusion
