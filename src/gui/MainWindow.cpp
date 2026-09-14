#include "gui/MainWindow.h"
#include "gui/OpenGLWidget.h"
#include "gui/MetricsPanel.h"
#include "gui/ControlPanel.h"

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

void MainWindow::onResetClicked() {
    const bool was_capturing = pipeline_->isRunning();

    pipeline_->reset();

    if (gl_widget_) gl_widget_->clearGeometry();
    control_panel_->setExportEnabled(false);

    if (was_capturing) {
        if (!pipeline_->start()) {
            QMessageBox::critical(this, "Error",
                "Scan was reset but failed to restart capture.\n"
                "Check the Kinect connection and press Start again.");
            control_panel_->onPipelineStopped();
            statusBar()->showMessage("Reset failed to restart — press Start.");
        } else {
            control_panel_->onPipelineStarted();
            statusBar()->showMessage("Scan reset — capturing again.");
        }
    } else {
        control_panel_->onPipelineStopped();
        statusBar()->showMessage("Volume cleared. Press Start to capture.");
    }

    if (metrics_panel_) metrics_panel_->update(pipeline_->metricsSnapshot());
}

void MainWindow::onExportPLY() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export PLY", "scan.ply", "PLY Files (*.ply)");
    if (path.isEmpty()) return;

    statusBar()->showMessage("Exporting PLY...");
    bool ok = pipeline_->exportPLY(path.toStdString());
    statusBar()->showMessage(ok ? ("PLY exported: " + path) : "PLY export FAILED.");
    if (!ok) {
        QMessageBox::warning(
            this,
            "Export Error",
            "PLY export failed.\nScan a little more so a mesh can be extracted, then try again.");
    }
}

void MainWindow::onExportGLB() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export GLB", "scan.glb", "GLB Files (*.glb)");
    if (path.isEmpty()) return;

    statusBar()->showMessage("Exporting GLB...");
    bool ok = pipeline_->exportGLB(path.toStdString());
    statusBar()->showMessage(ok ? ("GLB exported: " + path) : "GLB export FAILED.");
    if (!ok) {
        QMessageBox::warning(
            this,
            "Export Error",
            "GLB export failed.\nScan a little more so a mesh can be extracted, then try again.");
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
        control_panel_->setExportEnabled(true);
    }

}

void MainWindow::onMetricsTimer() {
    if (!pipeline_ || !metrics_panel_) return;
    app::PipelineMetrics m = pipeline_->metricsSnapshot();
    metrics_panel_->update(m);
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
