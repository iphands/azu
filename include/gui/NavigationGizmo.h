#pragma once

#include <QWidget>
#include <QPoint>

namespace kfusion {
namespace gui {

class NavigationGizmo : public QWidget {
    Q_OBJECT
public:
    explicit NavigationGizmo(QWidget* parent = nullptr);

signals:
    // Float degrees: an int signature rounded sub-degree drags away to nothing.
    void cameraRotationChanged(float pitch, float yaw, float roll);

public slots:
    void setCameraRotation(float pitch, float yaw, float roll);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    float pitch_ = 0.0f;
    float yaw_   = 0.0f;
    float roll_  = 0.0f;
    QPoint last_mouse_pos_;
    bool   dragging_ = false;
};

} // namespace gui
} // namespace kfusion
