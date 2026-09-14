#pragma once

#include <QMainWindow>
#include <QTimer>
#include <memory>
#include "app/PipelineController.h"

QT_BEGIN_NAMESPACE
class QSplitter;
class QLabel;
class QPushButton;
class QProgressBar;
class QComboBox;
class QStatusBar;
QT_END_NAMESPACE

namespace kfusion {
namespace gui {

class OpenGLWidget;
class MetricsPanel;
class ControlPanel;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(sensor::PreprocessBackend preferred_backend = sensor::PreprocessBackend::Auto,
                        QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onStartClicked();
    void onStopClicked();
    void onResetClicked();
    void onExportPLY();
    void onExportGLB();
    void onModeChanged(int index);
    void onFrameReady(const sensor::FrameData& frame);
    void onMeshReady();
    void onMetricsTimer();

private:
    std::unique_ptr<app::PipelineController> pipeline_;

    OpenGLWidget*  gl_widget_       = nullptr;
    MetricsPanel*  metrics_panel_   = nullptr;
    ControlPanel*  control_panel_   = nullptr;

    QTimer* metrics_timer_ = nullptr;

    Eigen::Vector3f cage_origin_{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f cage_size_{0.0f, 0.0f, 0.0f};
    // Amber fires only after N consecutive outside ticks, inset by margin so a
    // pose on the boundary face (identity start sits on the Z=0 default face)
    // doesn't trip; latch holds through TrackingLost until pose re-enters.
    float cage_exit_margin_ = 0.15f;
    int   cage_out_streak_  = 0;

    void setupUI();
    void connectSignals();
    void updateGlobalStyle();
    void applyVolumeCage(const app::FusionHyperparams& h);

    float ui_scale_ = 1.0f;
    void keyPressEvent(QKeyEvent* event) override;
};

} // namespace gui
} // namespace kfusion
