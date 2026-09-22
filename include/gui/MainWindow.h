#pragma once

#include <QMainWindow>
#include <QTimer>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include "app/PipelineController.h"

QT_BEGIN_NAMESPACE
class QSplitter;
class QLabel;
class QPushButton;
class QProgressBar;
class QComboBox;
class QStatusBar;
class QString;
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
    void updateUiStyle();
    void applyVolumeCage(const app::FusionHyperparams& h);

    // ---- background export/reset work (big-fix Todo 26) ----
    // At most ONE background operation runs at a time; busy_op_ is read and
    // written on the GUI thread only. Every worker thread is joined by
    // ~MainWindow BEFORE pipeline_ is stopped or freed, so a worker lambda
    // may safely hold a raw PipelineController pointer, and a completion
    // posted through QMetaObject::invokeMethod(this, ..., Qt::QueuedConnection)
    // can never run on a destroyed window (Qt drops the queued functor when
    // its context object dies first, and the join orders that destruction).
    enum class BackgroundOp { None, ExportPly, ExportGlb, Reset };

    /** Launch `work` on a worker thread; its bool result is delivered to
     *  finishBackgroundOp() on the GUI thread. Refuses overlapping ops. */
    void startBackgroundOp(BackgroundOp op, std::function<bool()> work);
    /** GUI-thread completion handler for a finished background op. */
    void finishBackgroundOp(BackgroundOp op, bool ok);
    /** Shared launch path for BOTH export buttons (runs exportPLY/exportGLB,
     *  which both route through PipelineController::exportMesh). */
    void startExport(BackgroundOp op, const QString& path);
    void joinBackgroundWorkers();

    BackgroundOp busy_op_ = BackgroundOp::None;
    bool reset_restart_capture_ = false; // Reset-op payload, GUI thread only
    bool mesh_available_ = false;        // export-enable mirror for restore
    std::vector<std::thread> background_workers_;

    float ui_scale_ = 1.0f;
    void keyPressEvent(QKeyEvent* event) override;
};

} // namespace gui
} // namespace kfusion
