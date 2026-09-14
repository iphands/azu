#include "gui/OpenGLWidget.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <QSurfaceFormat>

namespace kfusion {
namespace gui {

OpenGLWidget::OpenGLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    fmt.setSamples(4);
    setFormat(fmt);

    setMinimumSize(640, 480);
    setFocusPolicy(Qt::StrongFocus); // Required to receive keyboard events
    
    physics_timer_ = new QTimer(this);
    connect(physics_timer_, &QTimer::timeout, this, &OpenGLWidget::updatePhysics);
    physics_timer_->start(16); // ~60 fps
    frame_timer_.start();
}

OpenGLWidget::~OpenGLWidget() {
    makeCurrent();
    renderer_.reset();
    doneCurrent();
}

void OpenGLWidget::setRenderMode(rendering::RenderMode mode) {
    if (renderer_) renderer_->setMode(mode);
    update();
}

void OpenGLWidget::updatePointCloud(const sensor::FrameData& frame) {
    makeCurrent();
    if (renderer_) renderer_->uploadPointCloud(frame);
    doneCurrent();
    update();
}

void OpenGLWidget::updateMesh(const meshing::MeshData& mesh) {
    makeCurrent();
    if (renderer_) renderer_->uploadMesh(mesh);
    doneCurrent();
    update();
}

void OpenGLWidget::clearGeometry() {
    makeCurrent();
    if (renderer_) renderer_->clearGeometry();
    doneCurrent();
    update();
}

void OpenGLWidget::setVolumeBox(const Eigen::Vector3f& origin, const Eigen::Vector3f& size) {
    cage_origin_ = origin;
    cage_size_   = size;
    if (renderer_) {
        renderer_->setVolumeBox(origin, size);
        update();
    }
}

void OpenGLWidget::setVolumeBoxVisible(bool visible) {
    cage_visible_ = visible;
    if (renderer_) {
        renderer_->setVolumeBoxVisible(visible);
        update();
    }
}

void OpenGLWidget::setVolumeBoxOutside(bool outside) {
    if (outside == cage_outside_) return;  // 5 Hz poller: repaint only on real transitions
    cage_outside_ = outside;
    if (renderer_) {
        renderer_->setVolumeBoxOutside(outside);
        update();
    }
}

void OpenGLWidget::setCameraRotation(int pitch, int yaw, int roll) {
    if (!renderer_) return;
    auto& cam = renderer_->camera();
    cam.setElevation(-pitch * M_PI / 180.0f); // Map Pitch to elevation
    cam.setAzimuth(-yaw * M_PI / 180.0f);     // Map Yaw to azimuth
    cam.setRoll(roll * M_PI / 180.0f);        // Map Roll
    update();
}

void OpenGLWidget::initializeGL() {
    initializeOpenGLFunctions();
    renderer_ = std::make_unique<rendering::PreviewRenderer>();
    renderer_->initialize();
    renderer_->setVolumeBox(cage_origin_, cage_size_);
    renderer_->setVolumeBoxVisible(cage_visible_);
    renderer_->setVolumeBoxOutside(cage_outside_);
}

void OpenGLWidget::resizeGL(int w, int h) {
    if (renderer_) renderer_->resize(w, h);
}

void OpenGLWidget::paintGL() {
    if (renderer_) renderer_->render();
}

void OpenGLWidget::mousePressEvent(QMouseEvent* e) {
    last_mouse_pos_  = e->pos();
    mouse_pressed_   = true;
    pressed_button_  = e->button();
}

void OpenGLWidget::mouseMoveEvent(QMouseEvent* e) {
    if (!mouse_pressed_ || !renderer_) return;

    QPoint delta = e->pos() - last_mouse_pos_;
    last_mouse_pos_ = e->pos();

    if (pressed_button_ == Qt::LeftButton) {
        renderer_->camera().rotate(
            static_cast<float>(delta.x()),
            static_cast<float>(delta.y())
        );
    } else if (pressed_button_ == Qt::RightButton) {
        float dx = static_cast<float>(delta.x());
        float dy = static_cast<float>(delta.y());
        
        // Axis constraint logic
        if (axis_lock_x_ && !axis_lock_y_ && !axis_lock_z_) {
            dy = 0.0f; // Only move along X
        } else if (axis_lock_y_ && !axis_lock_x_ && !axis_lock_z_) {
            dx = 0.0f; // Only move along Y
        } else if (axis_lock_z_ && !axis_lock_x_ && !axis_lock_y_) {
            // Z panning: move along world Z
            renderer_->camera().move(Eigen::Vector3f(0.0f, 0.0f, 1.0f), -dy * 0.005f);
            dx = 0.0f; dy = 0.0f;
        }

        if (dx != 0.0f || dy != 0.0f) {
            renderer_->camera().pan(dx, dy);
        }
    }
    
    // Emit rotation back to sliders
    auto& cam = renderer_->camera();
    emit cameraRotated(
        -cam.elevation() * 180.0f / M_PI,
        -cam.azimuth() * 180.0f / M_PI,
        cam.roll() * 180.0f / M_PI
    );
    
    update();
}

void OpenGLWidget::wheelEvent(QWheelEvent* e) {
    if (!renderer_) return;
    float delta = static_cast<float>(e->angleDelta().y()) / 120.0f;
    renderer_->camera().zoom(delta);
    update();
}

void OpenGLWidget::keyPressEvent(QKeyEvent* e) {
    pressed_keys_.insert(e->key());

    if (!renderer_) return;
    
    // Toggle axis locks
    if (e->key() == Qt::Key_X) axis_lock_x_ = true;
    if (e->key() == Qt::Key_Y) axis_lock_y_ = true;
    if (e->key() == Qt::Key_Z) axis_lock_z_ = true;

    // Toggle camera mode
    if (e->key() == Qt::Key_Tab) {
        auto& cam = renderer_->camera();
        if (cam.mode() == rendering::Camera::Mode::Orbit) {
            cam.setMode(rendering::Camera::Mode::Free);
        } else {
            cam.setMode(rendering::Camera::Mode::Orbit);
        }
        update();
    }
    
    // Focus on target (center of scene)
    if (e->key() == Qt::Key_F) {
        auto& cam = renderer_->camera();
        cam.setMode(rendering::Camera::Mode::Orbit);
        // Default origin look
        cam.setTarget(Eigen::Vector3f(0.0f, 0.0f, 1.0f)); 
        update();
    }
}

void OpenGLWidget::keyReleaseEvent(QKeyEvent* e) {
    pressed_keys_.erase(e->key());
    if (e->key() == Qt::Key_X) axis_lock_x_ = false;
    if (e->key() == Qt::Key_Y) axis_lock_y_ = false;
    if (e->key() == Qt::Key_Z) axis_lock_z_ = false;
}

void OpenGLWidget::focusOutEvent(QFocusEvent* e) {
    QOpenGLWidget::focusOutEvent(e);
    pressed_keys_.clear();
    axis_lock_x_ = false;
    axis_lock_y_ = false;
    axis_lock_z_ = false;
}

void OpenGLWidget::updatePhysics() {
    float dt = static_cast<float>(frame_timer_.restart()) / 1000.0f;
    if (!renderer_) return;

    float speed = 2.0f; // meters per second
    if (pressed_keys_.count(Qt::Key_Shift)) speed *= 3.0f;

    Eigen::Vector3f move_dir(0, 0, 0);

    if (pressed_keys_.count(Qt::Key_W)) move_dir.z() += 1.0f;
    if (pressed_keys_.count(Qt::Key_S)) move_dir.z() -= 1.0f;
    if (pressed_keys_.count(Qt::Key_A)) move_dir.x() -= 1.0f;
    if (pressed_keys_.count(Qt::Key_D)) move_dir.x() += 1.0f;
    if (pressed_keys_.count(Qt::Key_Q)) move_dir.y() -= 1.0f;
    if (pressed_keys_.count(Qt::Key_E)) move_dir.y() += 1.0f;

    if (move_dir.squaredNorm() > 0.001f) {
        move_dir.normalize();
        renderer_->camera().move(move_dir, speed * dt);
        update();

        // Sync sliders
        auto& cam = renderer_->camera();
        emit cameraRotated(
            -cam.elevation() * 180.0f / M_PI,
            -cam.azimuth() * 180.0f / M_PI,
            cam.roll() * 180.0f / M_PI
        );
    }
}

} // namespace gui
} // namespace kfusion
