#include "rendering/PreviewRenderer.h"
#include <Eigen/Core>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace kfusion {
namespace rendering {

namespace {

// Kinect camera space: +X right, +Y down (image), +Z forward. GL view: +Y up, −Z forward.
// Equivalent to 180° about X: (x,y,z) → (x,−y,−z). Keeps right-handed volume.
Eigen::Matrix4f kinectToOpenGL() {
    Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
    F(1, 1) = -1.0f;
    F(2, 2) = -1.0f;
    return F;
}

Eigen::Matrix4f sensorProjection(int viewport_w, int viewport_h,
                                 float near_z = 0.05f,
                                 float far_z = 10.0f) {
    const float sx = static_cast<float>(viewport_w) / static_cast<float>(sensor::FRAME_W);
    const float sy = static_cast<float>(viewport_h) / static_cast<float>(sensor::FRAME_H);
    const float fx = static_cast<float>(sensor::FX) * sx;
    const float fy = static_cast<float>(sensor::FY) * sy;
    const float cx = (static_cast<float>(sensor::CX) + 0.5f) * sx;
    const float cy = (static_cast<float>(sensor::CY) + 0.5f) * sy;
    const float w  = static_cast<float>(viewport_w);
    const float h  = static_cast<float>(viewport_h);

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0, 0) = 2.0f * fx / w;
    P(1, 1) = 2.0f * fy / h;
    P(0, 2) = 1.0f - 2.0f * cx / w;
    P(1, 2) = 2.0f * cy / h - 1.0f;
    P(2, 2) = -(far_z + near_z) / (far_z - near_z);
    P(2, 3) = -(2.0f * far_z * near_z) / (far_z - near_z);
    P(3, 2) = -1.0f;
    return P;
}

} // namespace

// ---------------------------------------------------------------------------
// Shader sources
// ---------------------------------------------------------------------------

const char* PreviewRenderer::POINTCLOUD_VERT = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uMVP;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = 1.5;
    vColor = aColor;
}
)glsl";

const char* PreviewRenderer::POINTCLOUD_FRAG = R"glsl(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)glsl";

const char* PreviewRenderer::MESH_VERT = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uMVP;
uniform mat4 uModelView;
uniform mat3 uNormalMatrix;
out vec3 vNormal;
out vec3 vFragPos;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos = vec3(uModelView * vec4(aPos, 1.0));
    vNormal  = normalize(uNormalMatrix * aNormal);
    vColor   = aColor;
}
)glsl";

const char* PreviewRenderer::MESH_FRAG = R"glsl(
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;
in vec3 vColor;
out vec4 FragColor;
void main() {
    vec3 lightDir = normalize(vec3(1.0, 2.0, 3.0));
    float diff = max(dot(vNormal, lightDir), 0.0);
    vec3 ambient = vec3(0.15);
    vec3 diffuse = vec3(0.6) * diff;
    float spec_k = pow(max(dot(normalize(-vFragPos), reflect(-lightDir, vNormal)), 0.0), 32.0);
    vec3 specular = vec3(0.2) * spec_k;
    FragColor = vec4((ambient + diffuse + specular) * vColor, 1.0);
}
)glsl";

const char* PreviewRenderer::CAGE_VERT = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)glsl";

const char* PreviewRenderer::CAGE_FRAG = R"glsl(
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
)glsl";

// ---------------------------------------------------------------------------

PreviewRenderer::PreviewRenderer() = default;

PreviewRenderer::~PreviewRenderer() {
    if (!initialized_) return;
    if (pc_vao_) { glDeleteVertexArrays(1, &pc_vao_); }
    if (pc_vbo_pos_) { glDeleteBuffers(1, &pc_vbo_pos_); }
    if (pc_vbo_col_) { glDeleteBuffers(1, &pc_vbo_col_); }
    if (mesh_vao_) { glDeleteVertexArrays(1, &mesh_vao_); }
    if (mesh_vbo_pos_) { glDeleteBuffers(1, &mesh_vbo_pos_); }
    if (mesh_vbo_norm_) { glDeleteBuffers(1, &mesh_vbo_norm_); }
    if (mesh_ebo_) { glDeleteBuffers(1, &mesh_ebo_); }
    if (cage_vao_) { glDeleteVertexArrays(1, &cage_vao_); }
    if (cage_vbo_) { glDeleteBuffers(1, &cage_vbo_); }
}

void PreviewRenderer::initialize() {
    initializeOpenGLFunctions();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.08f, 0.09f, 0.11f, 1.0f);

    pc_shader_.load(POINTCLOUD_VERT, POINTCLOUD_FRAG);
    mesh_shader_.load(MESH_VERT, MESH_FRAG);
    cage_shader_.load(CAGE_VERT, CAGE_FRAG);

    initPointCloudBuffers();
    initMeshBuffers();
    initCageBuffers();

    GLfloat lw_range[2] = {1.0f, 1.0f};
    glGetFloatv(GL_LINE_WIDTH_RANGE, lw_range);
    // Clamp to avoid INVALID_VALUE; core-profile drivers commonly report [1,1],
    // so the cage usually stays 1px — real thick outlines need geometry shaders.
    cage_line_width_ = std::min(2.0f, lw_range[1]);

    initialized_ = true;
}

void PreviewRenderer::resize(int w, int h) {
    viewport_w_ = std::max(w, 1);
    viewport_h_ = std::max(h, 1);
    glViewport(0, 0, viewport_w_, viewport_h_);
}

void PreviewRenderer::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_PROGRAM_POINT_SIZE);

    if (mode_ == RenderMode::PointCloud) {
        renderPointCloud();
    } else if (mesh_index_count_ > 0) {
        renderMesh();
    } else {
        // Mesh mode selected but no extraction yet (or empty) — keep showing live depth cloud.
        renderPointCloud();
    }

    renderCage();
}

void PreviewRenderer::initPointCloudBuffers() {
    glGenVertexArrays(1, &pc_vao_);
    glGenBuffers(1, &pc_vbo_pos_);
    glGenBuffers(1, &pc_vbo_col_);

    glBindVertexArray(pc_vao_);

    glBindBuffer(GL_ARRAY_BUFFER, pc_vbo_pos_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, pc_vbo_col_);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void PreviewRenderer::initMeshBuffers() {
    glGenVertexArrays(1, &mesh_vao_);
    glGenBuffers(1, &mesh_vbo_pos_);
    glGenBuffers(1, &mesh_vbo_norm_);
    glGenBuffers(1, &mesh_vbo_col_);
    glGenBuffers(1, &mesh_ebo_);

    glBindVertexArray(mesh_vao_);

    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_pos_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_norm_);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_col_);
    glVertexAttribPointer(2, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, nullptr);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ebo_);

    glBindVertexArray(0);
}

void PreviewRenderer::uploadPointCloud(const sensor::FrameData& frame) {
    if (!initialized_) return;

    const int N = frame.width * frame.height;
    std::vector<float> positions, colors;
    positions.reserve(N * 3);
    colors.reserve(N * 3);
    int count = 0;

    for (int i = 0; i < N; ++i) {
        // Handle both live frames (depth check) and raycasted model frames (norm check)
        // Raycasted points are in WORLD space, so z > 0 check is invalid. 
        // Use norm > 1e-6 to identify valid hits and skip empty points (0,0,0).
        const auto& v = frame.vertices[i];
        bool valid = (frame.depth_meters.size() > i) ? (frame.depth_meters[i] > 0.0f) : (v.norm() > 1e-6f);
        if (!valid || std::isnan(v.x()) || std::isnan(v.y()) || std::isnan(v.z())) continue;

        positions.push_back(v.x());
        positions.push_back(v.y());
        positions.push_back(v.z());
        colors.push_back(frame.rgb[i*3+0] / 255.0f);
        colors.push_back(frame.rgb[i*3+1] / 255.0f);
        colors.push_back(frame.rgb[i*3+2] / 255.0f);
        ++count;
    }

    pc_count_ = count;

    glBindBuffer(GL_ARRAY_BUFFER, pc_vbo_pos_);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float),
                 positions.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, pc_vbo_col_);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float),
                 colors.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PreviewRenderer::clearGeometry() {
    pc_count_         = 0;
    mesh_index_count_ = 0;
}

void PreviewRenderer::uploadMesh(const meshing::MeshData& mesh) {
    if (!initialized_ || mesh.empty()) return;

    glBindVertexArray(mesh_vao_);

    // Positions
    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_pos_);
    glBufferData(GL_ARRAY_BUFFER,
                 mesh.positions.size() * sizeof(Eigen::Vector3f),
                 mesh.positions.data(), GL_DYNAMIC_DRAW);

    // Normals
    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_norm_);
    glBufferData(GL_ARRAY_BUFFER,
                 mesh.normals.size() * sizeof(Eigen::Vector3f),
                 mesh.normals.data(), GL_DYNAMIC_DRAW);

    // Colors
    if (!mesh.colors.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_col_);
        glBufferData(GL_ARRAY_BUFFER,
                     mesh.colors.size(),
                     mesh.colors.data(), GL_DYNAMIC_DRAW);
    }

    // Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 mesh.indices.size() * sizeof(uint32_t),
                 mesh.indices.data(), GL_DYNAMIC_DRAW);

    mesh_index_count_ = static_cast<int>(mesh.indices.size());
    glBindVertexArray(0);
}

void PreviewRenderer::renderPointCloud() {
    if (!pc_shader_.isValid() || pc_count_ == 0) return;

    float aspect = static_cast<float>(viewport_w_) / static_cast<float>(viewport_h_);
    Eigen::Matrix4f V   = camera_.viewMatrix();
    Eigen::Matrix4f P   = camera_.projectionMatrix(aspect);
    const Eigen::Matrix4f F   = kinectToOpenGL();
    const Eigen::Matrix4f MVP = P * V * F;

    pc_shader_.use();
    pc_shader_.setUniformMat4("uMVP", MVP.data());

    glBindVertexArray(pc_vao_);
    glDrawArrays(GL_POINTS, 0, pc_count_);
    glBindVertexArray(0);

    pc_shader_.disuse();
}

void PreviewRenderer::renderMesh() {
    if (!mesh_shader_.isValid() || mesh_index_count_ == 0) return;

    float aspect = static_cast<float>(viewport_w_) / static_cast<float>(viewport_h_);
    Eigen::Matrix4f V   = camera_.viewMatrix();
    Eigen::Matrix4f P   = camera_.projectionMatrix(aspect);
    const Eigen::Matrix4f F    = kinectToOpenGL();
    const Eigen::Matrix4f MV   = V * F;
    const Eigen::Matrix4f MVP  = P * MV;

    mesh_shader_.use();
    mesh_shader_.setUniformMat4("uMVP", MVP.data());
    mesh_shader_.setUniformMat4("uModelView", MV.data());

    Eigen::Matrix3f normalMatrix = MV.block<3,3>(0,0).inverse().transpose();
    mesh_shader_.setUniformMat3("uNormalMatrix", normalMatrix.data());

    glBindVertexArray(mesh_vao_);
    glDrawElements(GL_TRIANGLES, mesh_index_count_, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    mesh_shader_.disuse();
}

void PreviewRenderer::setVolumeBox(const Eigen::Vector3f& origin, const Eigen::Vector3f& size) {
    cage_origin_ = origin;
    cage_size_   = size;
}

// 12 cube edges as line segments in unit space; scaled/translated by uMVP.
static const float CUBE_EDGES[24 * 3] = {
    0,0,0,  1,0,0,   1,0,0,  1,1,0,   1,1,0,  0,1,0,   0,1,0,  0,0,0,
    0,0,1,  1,0,1,   1,0,1,  1,1,1,   1,1,1,  0,1,1,   0,1,1,  0,0,1,
    0,0,0,  0,0,1,   1,0,0,  1,0,1,   1,1,0,  1,1,1,   0,1,0,  0,1,1,
};

void PreviewRenderer::initCageBuffers() {
    glGenVertexArrays(1, &cage_vao_);
    glGenBuffers(1, &cage_vbo_);
    glBindVertexArray(cage_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, cage_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(CUBE_EDGES), CUBE_EDGES, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindVertexArray(0);
}

void PreviewRenderer::renderCage() {
    if (!cage_visible_ || !cage_shader_.isValid() || cage_size_.minCoeff() <= 0.0f) return;

    float aspect = static_cast<float>(viewport_w_) / static_cast<float>(viewport_h_);
    Eigen::Matrix4f V = camera_.viewMatrix();
    Eigen::Matrix4f P = camera_.projectionMatrix(aspect);
    const Eigen::Matrix4f F = kinectToOpenGL();
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M(0,0) = cage_size_.x();  M(0,3) = cage_origin_.x();
    M(1,1) = cage_size_.y();  M(1,3) = cage_origin_.y();
    M(2,2) = cage_size_.z();  M(2,3) = cage_origin_.z();
    const Eigen::Matrix4f MVP = P * V * F * M;

    cage_shader_.use();
    cage_shader_.setUniformMat4("uMVP", MVP.data());
    if (cage_outside_) cage_shader_.setUniformVec3("uColor", 0.98f, 0.67f, 0.27f);
    else               cage_shader_.setUniformVec3("uColor", 0.353f, 0.624f, 1.0f);

    glDepthFunc(GL_LEQUAL);
    glLineWidth(cage_line_width_);
    glBindVertexArray(cage_vao_);
    glDrawArrays(GL_LINES, 0, 24);
    glBindVertexArray(0);
    glDepthFunc(GL_LESS);
    glLineWidth(1.0f);

    cage_shader_.disuse();
}

} // namespace rendering
} // namespace kfusion
