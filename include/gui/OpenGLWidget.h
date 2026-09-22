#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QPoint>
#include <QTimer>
#include <QElapsedTimer>
#include <QKeyEvent>
#include <memory>
#include <set>
#include "rendering/PreviewRenderer.h"
#include "meshing/MeshData.h"

namespace kfusion {
namespace gui {

class OpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT
public:
    explicit OpenGLWidget(QWidget* parent = nullptr);
    ~OpenGLWidget() override;

    void setRenderMode(rendering::RenderMode mode);
    void updatePointCloud(const sensor::FrameData& frame);
    void updateMesh(const meshing::MeshData& mesh);
    void clearGeometry();

    void setVolumeBox(const Eigen::Vector3f& origin, const Eigen::Vector3f& size);
    void setVolumeBoxVisible(bool visible);
    void setVolumeBoxOutside(bool outside);

signals:
    // Float degrees end-to-end: an int signature truncated sub-degree drags to 0.
    void cameraRotated(float pitch, float yaw, float roll);

public slots:
    void setCameraRotation(float pitch, float yaw, float roll);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
    void keyReleaseEvent(QKeyEvent* e) override;
    void focusOutEvent(QFocusEvent* e) override;

private slots:
    void updatePhysics();

private:
    void syncRotationFeedback();

    // The physics timer starts with this exact interval and
    // clampFrameDeltaSeconds() replaces the first frame with it, so the nominal
    // tick and the real tick rate can never drift apart.
    static constexpr float kPhysicsTickSeconds = 0.016f;

    std::unique_ptr<rendering::PreviewRenderer> renderer_;

    // True once initializeGL confirmed the live context clears the renderer's
    // desktop-3.3-Core floor. Gates the one-time "GL inadequate" paint warning so
    // a mis-capable host logs the reason instead of silently showing a black box.
    bool gl_adequate_ = false;

    // Volume-cage state cache: MainWindow pushes the box before initializeGL
    // creates renderer_; flush to renderer_ at creation time.
    Eigen::Vector3f cage_origin_{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f cage_size_{0.0f, 0.0f, 0.0f};
    bool cage_visible_ = true;
    bool cage_outside_ = false;
    QPoint last_mouse_pos_;
    bool   mouse_pressed_ = false;
    Qt::MouseButton pressed_button_ = Qt::NoButton;
    
    QTimer* physics_timer_;
    QElapsedTimer frame_timer_;
    bool first_physics_tick_ = true;
    std::set<int> pressed_keys_;
    
    // Axis constraints for panning
    bool axis_lock_x_ = false;
    bool axis_lock_y_ = false;
    bool axis_lock_z_ = false;
};

} // namespace gui
} // namespace kfusion
