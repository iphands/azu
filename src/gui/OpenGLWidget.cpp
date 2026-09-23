#include "gui/OpenGLWidget.h"
#include <QThread>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QOpenGLContext>
#include <QSurfaceFormat>
#include <cstdio>
#include <string>
#include "gui/GlFormatQt.h"
#include "utils/Logger.h"

namespace kfusion {
namespace gui {

namespace {

// Radians live inside Camera, degrees live in the UI. Both conversions are
// float so no rotation-feedback path narrows through double or truncates
// through int on its way between the viewport and the panel.
constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
constexpr double kNanoPerSecond = 1.0 / 1000000000.0;

} // namespace

OpenGLWidget::OpenGLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    // The surface format is inherited from QSurfaceFormat::defaultFormat(), which
    // main() negotiated against the ladder (3.3 Core, MSAA dropped only if the
    // host could not satisfy it). Forcing samples(4) here re-introduced the
    // EGL_BAD_MATCH this fix removes, so the widget deliberately does NOT set its
    // own format; initializeGL validates whatever context actually resulted.

    setMinimumSize(640, 480);
    setFocusPolicy(Qt::StrongFocus); // Required to receive keyboard events
    
    physics_timer_ = new QTimer(this);
    connect(physics_timer_, &QTimer::timeout, this, &OpenGLWidget::updatePhysics);
    physics_timer_->start(static_cast<int>(kPhysicsTickSeconds * 1000.0f)); // ~60 fps
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
    // GL context work is GUI-thread only; pipeline workers must marshal here.
    Q_ASSERT(QThread::currentThread() == thread());
    makeCurrent();
    if (renderer_) renderer_->uploadPointCloud(frame);
    doneCurrent();
    update();
}

void OpenGLWidget::updateMesh(const meshing::MeshData& mesh) {
    // GL context work is GUI-thread only; pipeline workers must marshal here.
    Q_ASSERT(QThread::currentThread() == thread());
    makeCurrent();
    if (renderer_) renderer_->uploadMesh(mesh);
    doneCurrent();
    update();
}

void OpenGLWidget::clearGeometry() {
    // GL context work is GUI-thread only; pipeline workers must marshal here.
    Q_ASSERT(QThread::currentThread() == thread());
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

void OpenGLWidget::setCameraRotation(float pitch, float yaw, float roll) {
    if (!renderer_) return;
    auto& cam = renderer_->camera();
    cam.setElevation(-pitch * kDegToRad); // Map Pitch to elevation
    cam.setAzimuth(-yaw * kDegToRad);     // Map Yaw to azimuth
    cam.setRoll(roll * kDegToRad);        // Map Roll
    update();
}

void OpenGLWidget::initializeGL() {
    initializeOpenGLFunctions();

    // Validate the context Qt actually created against the renderer's hard floor
    // and log requested vs real, so a launch problem is legible from the log
    // alone. A context that dropped MSAA is a success (the ladder trades samples
    // away first); one below 3.3 Core, or an ES context, is named as a failure
    // because the shaders are '#version 330 core'.
    const auto requested_spec = kfusion::gui::toGlFormat(format());
    auto actual_spec = requested_spec;
    QOpenGLContext* ctx = context();
    if (ctx) {
        actual_spec = kfusion::gui::toGlFormat(ctx->format());
        const unsigned char* version = glGetString(GL_VERSION);
        int major = 0, minor = 0;
        if (version &&
            std::sscanf(reinterpret_cast<const char*>(version), "%d.%d", &major, &minor) == 2) {
            actual_spec.major = major;
            actual_spec.minor = minor;
        }
    } else {
        actual_spec.renderable = kfusion::gui::GlRenderableType::Unknown;
        actual_spec.profile = kfusion::gui::GlProfile::Unknown;
    }

    const auto gl_str = [this](GLenum name) -> std::string {
        const unsigned char* raw = glGetString(name);
        return raw ? std::string(reinterpret_cast<const char*>(raw)) : std::string("(null)");
    };

    const auto cmp = kfusion::gui::compareGlFormat(requested_spec, actual_spec);
    gl_adequate_ = cmp.adequate;

    KFLOG_INFO("gl", "OpenGLWidget requested " + kfusion::gui::describeGlFormat(requested_spec) +
                         " | actual " + kfusion::gui::describeGlFormat(actual_spec));
    KFLOG_INFO("gl", std::string("GL_VENDOR=") + gl_str(GL_VENDOR) + " | GL_RENDERER=" +
                         gl_str(GL_RENDERER) + " | GL_VERSION=" + gl_str(GL_VERSION) +
                         " | GL_SHADING_LANGUAGE_VERSION=" + gl_str(GL_SHADING_LANGUAGE_VERSION));

    if (!cmp.adequate) {
        std::string why;
        for (const std::string& reason : cmp.reasons) {
            why += reason;
            why += "; ";
        }
        KFLOG_ERROR("gl", "context does NOT clear the desktop 3.3 Core floor (" + why +
                              "): the preview shaders need '#version 330 core', so rendering "
                              "will fail on this context");
    } else if (!cmp.matches_request) {
        // Adequate but not what was asked for (typically MSAA down a rung): say
        // which rung was settled on, truthfully, without calling it an error.
        std::string note;
        for (const std::string& reason : cmp.reasons) {
            note += reason;
            note += "; ";
        }
        KFLOG_WARN("gl", "context adequate but downgraded from request: " + note);
    } else {
        KFLOG_INFO("gl", "context adequate and matches the requested format");
    }

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

void OpenGLWidget::syncRotationFeedback() {
    if (!renderer_) return;
    const auto& cam = renderer_->camera();
    // Radians inside Camera -> float degrees out; one place, no truncation.
    emit cameraRotated(-cam.elevation() * kRadToDeg,
                       -cam.azimuth()   * kRadToDeg,
                        cam.roll()      * kRadToDeg);
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
    syncRotationFeedback();

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
    // QElapsedTimer counts nanoseconds, so the tick has to be scaled by
    // 1e-9 to be seconds; the pre-fix /1000 fed microseconds to the
    // integrator and made free-flight move ~1000x too far.
    const double elapsed_seconds =
        static_cast<double>(frame_timer_.restart()) * kNanoPerSecond;
    const float dt = rendering::clampFrameDeltaSeconds(elapsed_seconds,
                                                      first_physics_tick_,
                                                      kPhysicsTickSeconds);
    first_physics_tick_ = false;
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
        syncRotationFeedback();
    }
}

} // namespace gui
} // namespace kfusion
