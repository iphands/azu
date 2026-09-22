#include "gui/NavigationGizmo.h"
#include <QPainter>
#include <QMouseEvent>
#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>

namespace kfusion {
namespace gui {

struct Vec3 { float x, y, z; };

static Vec3 rotate(Vec3 v, float pitch, float yaw, float roll) {
    float p = pitch * M_PI / 180.0f;
    float y = yaw * M_PI / 180.0f;
    float r = roll * M_PI / 180.0f;

    // Pitch around X
    float x1 = v.x;
    float y1 = v.y * cos(p) - v.z * sin(p);
    float z1 = v.y * sin(p) + v.z * cos(p);

    // Yaw around Y
    float x2 = x1 * cos(y) + z1 * sin(y);
    float y2 = y1;
    float z2 = -x1 * sin(y) + z1 * cos(y);

    // Roll around Z
    float x3 = x2 * cos(r) - y2 * sin(r);
    float y3 = x2 * sin(r) + y2 * cos(r);
    float z3 = z2;

    return {x3, y3, z3};
}

NavigationGizmo::NavigationGizmo(QWidget* parent) : QWidget(parent) {
    setMinimumSize(100, 100);
    setCursor(Qt::OpenHandCursor);
}

void NavigationGizmo::setCameraRotation(float pitch, float yaw, float roll) {
    pitch_ = pitch;
    yaw_ = yaw;
    roll_ = roll;
    update();
}

void NavigationGizmo::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        last_mouse_pos_ = event->pos();
        dragging_ = true;
        setCursor(Qt::ClosedHandCursor);
    }
}

void NavigationGizmo::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        QPoint delta = event->pos() - last_mouse_pos_;
        last_mouse_pos_ = event->pos();

        // Blender drag: dragging changes rotation, in float degrees
        yaw_   += static_cast<float>(delta.x());
        pitch_ += static_cast<float>(delta.y());

        emit cameraRotationChanged(pitch_, yaw_, roll_);
        update();
    }
}

void NavigationGizmo::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    dragging_ = false;
    // setCursor() overrides are sticky; nothing else resets the clenched fist.
    setCursor(Qt::OpenHandCursor);
}

void NavigationGizmo::enterEvent(QEnterEvent* event) {
    QWidget::enterEvent(event);
    if (!dragging_) setCursor(Qt::OpenHandCursor);
}

void NavigationGizmo::leaveEvent(QEvent* event) {
    QWidget::leaveEvent(event);
    // Qt keeps the mouse grab through a drag, so keep the fist until release.
    if (!dragging_) setCursor(Qt::OpenHandCursor);
}

void NavigationGizmo::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int side = qMin(width(), height());
    p.translate(width() / 2, height() / 2);
    
    float scale = side * 0.35f;

    // Background circle
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(40, 40, 40, 150));
    p.drawEllipse(QPointF(0, 0), side * 0.45, side * 0.45);

    // Calculate rotated endpoints
    Vec3 x_axis = rotate({1, 0, 0}, pitch_, yaw_, roll_);
    // Y-axis usually points up in 3D, but QPainter Y points down, so invert
    Vec3 y_axis = rotate({0, -1, 0}, pitch_, yaw_, roll_); 
    // Z-axis points towards screen usually, but let's invert for UI depth testing
    Vec3 z_axis = rotate({0, 0, -1}, pitch_, yaw_, roll_); 

    Vec3 nx_axis = {-x_axis.x, -x_axis.y, -x_axis.z};
    Vec3 ny_axis = {-y_axis.x, -y_axis.y, -y_axis.z};
    Vec3 nz_axis = {-z_axis.x, -z_axis.y, -z_axis.z};

    struct Element {
        float z;
        std::function<void()> draw;
    };

    std::vector<Element> elements;

    auto add_axis = [&](Vec3 v, QColor c, QString label, bool is_positive) {
        elements.push_back({v.z, [=, &p]() {
            QPointF end(v.x * scale, v.y * scale);
            if (is_positive) {
                p.setPen(QPen(c, 2.5, Qt::SolidLine, Qt::RoundCap));
                p.drawLine(QPointF(0, 0), end);
                
                p.setPen(Qt::NoPen);
                p.setBrush(c);
                p.drawEllipse(end, 8, 8);
                
                p.setPen(Qt::white);
                QFont font = p.font();
                font.setPixelSize(10);
                font.setBold(true);
                p.setFont(font);
                p.drawText(QRectF(end.x() - 8, end.y() - 8, 16, 16), Qt::AlignCenter, label);
            } else {
                p.setPen(Qt::NoPen);
                p.setBrush(c.darker(150));
                p.drawEllipse(end, 4, 4);
            }
        }});
    };

    // Blender colors
    add_axis(x_axis, QColor(255, 50, 80), "X", true);
    add_axis(y_axis, QColor(150, 220, 50), "Y", true);
    add_axis(z_axis, QColor(50, 150, 255), "Z", true);
    add_axis(nx_axis, QColor(255, 50, 80), "", false);
    add_axis(ny_axis, QColor(150, 220, 50), "", false);
    add_axis(nz_axis, QColor(50, 150, 255), "", false);

    // Sort by Z to draw back elements first
    std::sort(elements.begin(), elements.end(), [](const Element& a, const Element& b) {
        return a.z < b.z;
    });

    for (const auto& el : elements) {
        el.draw();
    }
}

} // namespace gui
} // namespace kfusion
