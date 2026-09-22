#include "gui/MainWindow.h"
#include "gui/OpenGLWidget.h"
#include "gui/MetricsPanel.h"
#include "gui/ControlPanel.h"

#include <Eigen/Core>

#include <QSplitter>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QCloseEvent>
#include <QStatusBar>
#include <QLabel>
#include <QApplication>

namespace kfusion {
namespace gui {

MainWindow::MainWindow(sensor::PreprocessBackend preferred_backend, QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("KinectFusionQt");
    resize(1600, 900);
    updateGlobalStyle();

    pipeline_ = std::make_unique<app::PipelineController>(preferred_backend);
    setupUI();
    connectSignals();

    // Timer for UI metrics refresh
    metrics_timer_ = new QTimer(this);
    connect(metrics_timer_, &QTimer::timeout, this, &MainWindow::onMetricsTimer);
    metrics_timer_->start(200); // 5 Hz UI refresh
}

MainWindow::~MainWindow() {
    // The join order here is load-bearing (Todo 26): worker threads call
    // controller methods through a captured raw PipelineController pointer,
    // and joining every worker while *this is still a live QObject is what
    // makes both that pointer and the workers' queued completion posts safe.
    // Only after the join may the controller be stopped/freed.
    joinBackgroundWorkers();
    if (pipeline_) pipeline_->stop();
}

void MainWindow::setupUI() {
    auto* central = new QWidget(this);
    setCentralWidget(central);

    auto* splitter = new QSplitter(Qt::Horizontal, central);
    splitter->setHandleWidth(3);
    splitter->setStyleSheet("QSplitter::handle { background: #333; }");

    auto* hbox = new QHBoxLayout(central);
    hbox->setSpacing(0);
    hbox->setContentsMargins(4, 4, 4, 4);
    hbox->addWidget(splitter);

    // Left: control panel
    control_panel_ = new ControlPanel(this);
    splitter->addWidget(control_panel_);

    // Center: OpenGL view
    gl_widget_ = new OpenGLWidget(this);
    splitter->addWidget(gl_widget_);

    // Right: metrics panel
    metrics_panel_ = new MetricsPanel(this);
    splitter->addWidget(metrics_panel_);

    // Set stretch factors: panels stay compact, OpenGL gets the space
    splitter->setStretchFactor(0, 0);  // control panel: don't stretch
    splitter->setStretchFactor(1, 1);  // OpenGL view: stretch to fill
    splitter->setStretchFactor(2, 0);  // metrics panel: don't stretch

    statusBar()->showMessage("Ready — Connect Kinect and press Start.");
}

void MainWindow::connectSignals() {
    // Control panel → pipeline
    connect(control_panel_, &ControlPanel::startClicked, this, &MainWindow::onStartClicked);
    connect(control_panel_, &ControlPanel::stopClicked,  this, &MainWindow::onStopClicked);
    connect(control_panel_, &ControlPanel::resetClicked, this, &MainWindow::onResetClicked);
    connect(control_panel_, &ControlPanel::exportPLYClicked, this, &MainWindow::onExportPLY);
    connect(control_panel_, &ControlPanel::exportGLBClicked, this, &MainWindow::onExportGLB);
    connect(control_panel_, &ControlPanel::modeChanged, this, &MainWindow::onModeChanged);
    connect(control_panel_, &ControlPanel::volumeCageToggled,
            gl_widget_, &OpenGLWidget::setVolumeBoxVisible);
    connect(control_panel_, &ControlPanel::threadsChanged, this, [this](int n) {
        if (pipeline_) pipeline_->setNumThreads(n);
    });

    connect(control_panel_, &ControlPanel::cameraRotationChanged,
            gl_widget_, &OpenGLWidget::setCameraRotation);
    connect(gl_widget_, &OpenGLWidget::cameraRotated,
            control_panel_, &ControlPanel::setCameraRotation);

    connect(control_panel_, &ControlPanel::hyperparamsApplyClicked, this, [this]() {
        auto h = control_panel_->hyperparamsFromUi();
        if (h.min_depth >= h.max_depth) {
            QMessageBox::warning(this, "Invalid depth range",
                "Depth minimum must be less than depth maximum.");
            return;
        }
        pipeline_->setHyperparams(h);
        control_panel_->setHyperparams(pipeline_->hyperparamsSnapshot());
        applyVolumeCage(pipeline_->hyperparamsSnapshot());
        statusBar()->showMessage("Hyperparameters applied.");
    });

    pipeline_->setFrameReadyCallback([this](const sensor::FrameData& frame) {
        onFrameReady(frame);
    });

    pipeline_->setMeshReadyCallback([this]() {
        QMetaObject::invokeMethod(this, [this]() {
            onMeshReady();
        }, Qt::QueuedConnection);
    });

    control_panel_->setHyperparams(pipeline_->hyperparamsSnapshot());
    applyVolumeCage(pipeline_->hyperparamsSnapshot());
}

void MainWindow::onStartClicked() {
    if (!pipeline_->start()) {
        QMessageBox::critical(this, "Error",
            "Failed to start pipeline.\n"
            "Ensure Kinect is connected and libfreenect is installed.\n"
            "You may need to run with appropriate device permissions (udev rules).");
        return;
    }
    control_panel_->onPipelineStarted();
    statusBar()->showMessage("Capturing...");
}

void MainWindow::onStopClicked() {
    pipeline_->stop();
    control_panel_->onPipelineStopped();
    statusBar()->showMessage("Stopped.");
}

void MainWindow::joinBackgroundWorkers() {
    for (auto& t : background_workers_) {
        if (t.joinable()) t.join();
    }
    background_workers_.clear();
}

void MainWindow::startBackgroundOp(BackgroundOp op, std::function<bool()> work) {
    if (busy_op_ != BackgroundOp::None) return;
    // Any previously finished worker is reaped here: its completion was
    // already delivered on this thread (busy_op_ is None), so its thread is
    // at most one postEvent call from exit and the join cannot stall.
    joinBackgroundWorkers();
    busy_op_ = op;
    background_workers_.emplace_back([this, op, work = std::move(work)]() mutable {
        bool ok = false;
        try {
            ok = work();
        } catch (...) {
            // A throwing controller call is a FAILED operation — never a
            // silent success and never a crash on a background thread.
            ok = false;
        }
        // Truthful completion, marshalled onto the GUI thread. `this` is
        // alive at this call because ~MainWindow joins every worker before
        // returning; and because the functor's context is `this`, Qt drops
        // it unexecuted if the window finishes destruction before the event
        // is processed. finishBackgroundOp therefore never runs on a dead
        // object and never fabricates a completion for an abandoned op.
        QMetaObject::invokeMethod(this, [this, op, ok]() {
            finishBackgroundOp(op, ok);
        }, Qt::QueuedConnection);
    });
}

void MainWindow::onResetClicked() {
    if (busy_op_ != BackgroundOp::None) return;
    const bool was_capturing = pipeline_->isRunning();

    // Truthful immediate state: the user asked to clear the volume, so the
    // viewport and export affordances go empty/disabled NOW; lifecycle
    // buttons latch off so nothing can overlap the reset while it runs.
    if (gl_widget_) gl_widget_->clearGeometry();
    mesh_available_ = false;
    control_panel_->setExportEnabled(false);
    control_panel_->setBusy(true);
    reset_restart_capture_ = was_capturing;
    statusBar()->showMessage("Resetting scan... (this runs in the background)");

    // reset() joins the pipeline workers and clears the volume; start()
    // reopens the sensor. Both are controller-thread-safe and touch no Qt —
    // exactly the work that must NOT run on the GUI thread.
    app::PipelineController* pipeline = pipeline_.get();
    startBackgroundOp(BackgroundOp::Reset, [pipeline, was_capturing]() {
        pipeline->reset();
        return !was_capturing || pipeline->start();
    });
}

void MainWindow::startExport(BackgroundOp op, const QString& path) {
    if (busy_op_ != BackgroundOp::None) return;
    const QString label = (op == BackgroundOp::ExportPly) ? "PLY" : "GLB";
    control_panel_->setExportEnabled(false);
    control_panel_->setBusy(true);
    statusBar()->showMessage("Exporting " + label + "... (this runs in the background)");

    app::PipelineController* pipeline = pipeline_.get();
    const std::string native_path = path.toStdString();
    const bool to_ply = (op == BackgroundOp::ExportPly);
    startBackgroundOp(op, [pipeline, to_ply, native_path]() {
        // exportPLY()/exportGLB() are thin wrappers over the shared
        // PipelineController::exportMesh(path, writer_fn) helper: mesh
        // acquisition (possibly one bounded extraction request) plus the
        // actual file write, all off the GUI thread.
        return to_ply ? pipeline->exportPLY(native_path)
                      : pipeline->exportGLB(native_path);
    });
}

void MainWindow::onExportPLY() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export PLY", "scan.ply", "PLY Files (*.ply)");
    if (path.isEmpty()) return;
    startExport(BackgroundOp::ExportPly, path);
}

void MainWindow::onExportGLB() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export GLB", "scan.glb", "GLB Files (*.glb)");
    if (path.isEmpty()) return;
    startExport(BackgroundOp::ExportGlb, path);
}

void MainWindow::finishBackgroundOp(BackgroundOp op, bool ok) {
    // Runs on the GUI thread (queued from the worker), after the controller
    // work is truly done. Restore every latched control from the REAL state,
    // not from an assumption: pipeline_->isRunning() and mesh_available_ are
    // re-read here so success and failure paths both land truthfully.
    busy_op_ = BackgroundOp::None;

    switch (op) {
    case BackgroundOp::ExportPly:
    case BackgroundOp::ExportGlb: {
        const QString label = (op == BackgroundOp::ExportPly) ? "PLY" : "GLB";
        control_panel_->setBusy(false);
        if (pipeline_->isRunning()) control_panel_->onPipelineStarted();
        else                        control_panel_->onPipelineStopped();
        control_panel_->setExportEnabled(mesh_available_);
        if (ok) {
            statusBar()->showMessage(label + " export complete.");
        } else {
            statusBar()->showMessage(label + " export FAILED.");
            QMessageBox::warning(
                this, "Export Error",
                label + " export failed.\nScan a little more so a mesh can be extracted, then try again.");
        }
        break;
    }
    case BackgroundOp::Reset: {
        const bool was_capturing = reset_restart_capture_;
        control_panel_->setBusy(false);
        if (pipeline_->isRunning()) control_panel_->onPipelineStarted();
        else                        control_panel_->onPipelineStopped();
        control_panel_->setExportEnabled(mesh_available_);
        if (was_capturing && !ok) {
            QMessageBox::critical(this, "Error",
                "Scan was reset but failed to restart capture.\n"
                "Check the Kinect connection and press Start again.");
            statusBar()->showMessage("Reset failed to restart — press Start.");
        } else if (was_capturing) {
            statusBar()->showMessage("Scan reset — capturing again.");
        } else {
            statusBar()->showMessage("Volume cleared. Press Start to capture.");
        }
        if (metrics_panel_) metrics_panel_->update(pipeline_->metricsSnapshot());
        break;
    }
    case BackgroundOp::None:
        break;
    }
}

void MainWindow::onModeChanged(int index) {
    rendering::RenderMode mode = (index == 0)
        ? rendering::RenderMode::PointCloud
        : rendering::RenderMode::Mesh;
    if (gl_widget_) gl_widget_->setRenderMode(mode);
}

void MainWindow::onFrameReady(const sensor::FrameData& frame) {
    if (gl_widget_) gl_widget_->updatePointCloud(frame);
}

void MainWindow::onMeshReady() {
    // Snapshot mesh from shared
    uint64_t ver;
    auto mesh = pipeline_->sharedMesh().snapshot(ver);

    if (gl_widget_ && mesh) {
        gl_widget_->updateMesh(*mesh);
        mesh_available_ = true;
        // A mesh can land WHILE a background export/reset owns the buttons;
        // re-enabling mid-op would lie about the latched state, so the
        // availability is recorded and finishBackgroundOp() replays it.
        if (busy_op_ == BackgroundOp::None) control_panel_->setExportEnabled(true);
    }

}

void MainWindow::onMetricsTimer() {
    if (!pipeline_ || !metrics_panel_) return;
    app::PipelineMetrics m = pipeline_->metricsSnapshot();
    metrics_panel_->update(m);

    const Eigen::Vector3f pos = pipeline_->currentPose().block<3, 1>(0, 3);
    const Eigen::Vector3f margin = Eigen::Vector3f::Constant(cage_exit_margin_);
    const bool inside = (pos.array() >= (cage_origin_ - margin).array()).all()
                     && (pos.array() <= (cage_origin_ + cage_size_ + margin).array()).all();
    if (m.state == app::PipelineState::Running)
        cage_out_streak_ = inside ? 0 : (cage_out_streak_ < 99 ? cage_out_streak_ + 1 : 99);
    if (gl_widget_)
        gl_widget_->setVolumeBoxOutside(cage_out_streak_ >= 3);
}

void MainWindow::applyVolumeCage(const app::FusionHyperparams& h) {
    cage_origin_ = h.tsdf.origin;
    const float extent = static_cast<float>(h.tsdf.resolution) * h.tsdf.voxel_size;
    cage_size_ = Eigen::Vector3f(extent, extent, extent);
    if (gl_widget_) gl_widget_->setVolumeBox(cage_origin_, cage_size_);
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (pipeline_) pipeline_->stop();
    event->accept();
}

void MainWindow::updateGlobalStyle() {
    int base_font     = 12;
    int title_font    = 14;
    int status_font   = 24;
    int button_font   = 16;
    
    int scaled_base   = static_cast<int>(base_font   * ui_scale_);
    int scaled_title  = static_cast<int>(title_font  * ui_scale_);
    int scaled_status = static_cast<int>(status_font * ui_scale_);
    int scaled_button = static_cast<int>(button_font * ui_scale_);

    QString style = QString(R"css(
        QMainWindow, QWidget { background-color: #1e1f24; color: #ddd; font-size: %1px; }
        QGroupBox {
            border: 2px solid #333;
            border-radius: 8px;
            margin-top: %2px;
            padding: %3px;
            font-weight: bold;
            color: #5a9fff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background: #3a3b42;
            border: 1px solid #555;
            border-radius: 6px;
            padding: %4px %5px;
            font-size: %6px;
            font-weight: bold;
            color: #eee;
        }
        QPushButton:hover { background: #4a4b52; border-color: #777; }
        QPushButton:pressed { background: #2a2b30; }
        QPushButton:disabled { color: #666; background: #2a2b30; }

        .ActionBtn { background: #3a7fcf; border: none; }
        .ActionBtn:hover { background: #4a8fdf; }
        .StopBtn { background: #cf3a3a; }
        .StopBtn:hover { background: #df4a4a; }

        QProgressBar { border: 1px solid #555; border-radius: 4px; background: #222; text-align: center; }
        QProgressBar::chunk { background: #3a7fcf; border-radius: 2px; }

        QComboBox, QSpinBox, QDoubleSpinBox {
            background: #2a2b30;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px 8px;
            min-height: %7px;
            color: #eee;
        }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #3a7fcf; }

        QLabel { color: #ccc; }
        .StatusLabel { font-size: %8px; font-weight: bold; }
        .PanelTitle { font-size: %9px; font-weight: bold; color: #eee; }

        QLabel[status="ok"] { color: #4f4; }
        QLabel[status="lost"] { color: #f44; }
        QLabel[status="running"] { color: #4f4; }
        QLabel[status="stopped"] { color: #fa4; }

        QLabel[overlap="good"] { color: #4f4; }
        QLabel[overlap="warn"] { color: #fa4; }
        QLabel[overlap="bad"] { color: #f44; }
        QLabel[overlap="warming"] { color: #888; }

        QScrollBar:vertical { background: #1e1f24; width: 10px; }
        QScrollBar::handle:vertical { background: #444; border-radius: 5px; min-height: 20px; }
        QScrollBar::add-line, QScrollBar::sub-line { background: none; }
    )css")
    .arg(scaled_base)
    .arg(static_cast<int>(15 * ui_scale_))
    .arg(static_cast<int>(10 * ui_scale_))
    .arg(static_cast<int>(10 * ui_scale_))
    .arg(static_cast<int>(15 * ui_scale_))
    .arg(scaled_button)
    .arg(static_cast<int>(24 * ui_scale_))
    .arg(scaled_status)
    .arg(scaled_title);

    qApp->setStyleSheet(style);
}

void MainWindow::keyPressEvent(QKeyEvent* event) {
    if (event->modifiers() & Qt::ControlModifier) {
        if (event->key() == Qt::Key_Plus || event->key() == Qt::Key_Equal) {
            ui_scale_ = std::min(3.0f, ui_scale_ + 0.1f);
            updateGlobalStyle();
            event->accept();
            return;
        } else if (event->key() == Qt::Key_Minus) {
            ui_scale_ = std::max(0.5f, ui_scale_ - 0.1f);
            updateGlobalStyle();
            event->accept();
            return;
        } else if (event->key() == Qt::Key_0) {
            ui_scale_ = 1.0f;
            updateGlobalStyle();
            event->accept();
            return;
        }
    }
    QMainWindow::keyPressEvent(event);
}

} // namespace gui
} // namespace kfusion
