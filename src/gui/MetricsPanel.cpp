#include "gui/MetricsPanel.h"
#include <QVBoxLayout>
#include <QGroupBox>
#include <QGridLayout>
#include <QString>
#include <QStyle>
#include <QVariant>

namespace kfusion {
namespace gui {

namespace {

// Style property name the MainWindow sheet keys on. "unknown" is the never-
// measured state (text "--"); it renders in the same neutral grey the old code
// used for a not-yet-warm panel, but stays a DISTINCT key so the band state
// machine never confuses "nothing measured yet" with "measured, still warming".
const char* overlapBandProperty(MetricsBand band) {
    switch (band) {
    case MetricsBand::kGood:    return "good";
    case MetricsBand::kWarn:    return "warn";
    case MetricsBand::kBad:     return "bad";
    case MetricsBand::kWarming: return "warming";
    case MetricsBand::kUnknown: return "unknown";
    }
    return "unknown";
}

} // namespace

MetricsPanel::MetricsPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
    setMinimumWidth(240);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
}

QLabel* MetricsPanel::makeLabel(const QString& text) {
    auto* lbl = new QLabel(text, this);
    return lbl;
}

void MetricsPanel::setupUI() {
    auto* root = new QVBoxLayout(this);
    root->setSpacing(10);
    root->setContentsMargins(10, 10, 10, 10);

    auto* title = new QLabel("Pipeline Metrics", this);
    title->setProperty("class", "PanelTitle");
    root->addWidget(title);

    // Performance group
    auto* grp_perf = new QGroupBox("Performance", this);
    auto* g_perf = new QGridLayout(grp_perf);
    g_perf->addWidget(new QLabel("Capture FPS:", this),  0, 0);
    lbl_fps_capture_  = makeLabel("--");
    g_perf->addWidget(lbl_fps_capture_, 0, 1);

    g_perf->addWidget(new QLabel("Tracking FPS:", this), 1, 0);
    lbl_fps_tracking_ = makeLabel("--");
    g_perf->addWidget(lbl_fps_tracking_, 1, 1);

    g_perf->addWidget(new QLabel("Frames:", this),       2, 0);
    lbl_frame_count_  = makeLabel("0");
    g_perf->addWidget(lbl_frame_count_, 2, 1);

    g_perf->addWidget(new QLabel("Backend:", this),      3, 0);
    lbl_backend_ = makeLabel("--");
    lbl_backend_->setWordWrap(true);
    g_perf->addWidget(lbl_backend_, 3, 1);
    root->addWidget(grp_perf);

    // Tracking group
    auto* grp_track = new QGroupBox("Tracking", this);
    auto* g_track = new QGridLayout(grp_track);
    g_track->addWidget(new QLabel("Status:", this), 0, 0);
    lbl_tracking_status_ = makeLabel("--");
    lbl_tracking_status_->setProperty("class", "StatusLabel");
    g_track->addWidget(lbl_tracking_status_, 0, 1);

    g_track->addWidget(new QLabel("ICP Error:", this), 1, 0);
    lbl_icp_error_ = makeLabel("--");
    g_track->addWidget(lbl_icp_error_, 1, 1);

    g_track->addWidget(new QLabel("Overlap:", this), 2, 0);
    lbl_icp_overlap_ = makeLabel("--");
    lbl_icp_overlap_->setProperty("class", "StatusLabel");
    lbl_icp_overlap_->setToolTip(tr(
        "Share of live depth points that matched the model.\n"
        ">40% healthy · 10–40% losing grip · <10% tracking will drop.\n"
        "Recovering: slow down, re-aim at geometry you already scanned, and stay inside\n"
        "the blue capture-volume box."));
    g_track->addWidget(lbl_icp_overlap_, 2, 1);
    root->addWidget(grp_track);

    // Volume group
    auto* grp_vol = new QGroupBox("Volume", this);
    auto* g_vol = new QVBoxLayout(grp_vol);
    auto* g_vol_grid = new QGridLayout();
    g_vol_grid->addWidget(new QLabel("Integrated:", this), 0, 0);
    lbl_integrated_frames_ = makeLabel("0");
    g_vol_grid->addWidget(lbl_integrated_frames_, 0, 1);

    g_vol_grid->addWidget(new QLabel("Usage:", this), 1, 0);
    lbl_volume_usage_ = makeLabel("0%");
    g_vol_grid->addWidget(lbl_volume_usage_, 1, 1);
    g_vol->addLayout(g_vol_grid);

    bar_volume_ = new QProgressBar(this);
    bar_volume_->setRange(0, 100);
    bar_volume_->setValue(0);
    bar_volume_->setTextVisible(false);
    bar_volume_->setFixedHeight(8);
    g_vol->addWidget(bar_volume_);
    root->addWidget(grp_vol);

    // Mesh group
    auto* grp_mesh = new QGroupBox("Mesh", this);
    auto* g_mesh = new QVBoxLayout(grp_mesh);
    auto* g_mesh_grid = new QGridLayout();
    g_mesh_grid->addWidget(new QLabel("Triangles:", this), 0, 0);
    lbl_mesh_triangles_ = makeLabel("0");
    g_mesh_grid->addWidget(lbl_mesh_triangles_, 0, 1);
    g_mesh->addLayout(g_mesh_grid);

    bar_mesh_extract_ = new QProgressBar(this);
    bar_mesh_extract_->setRange(0, 100);
    bar_mesh_extract_->setValue(0);
    bar_mesh_extract_->setTextVisible(false);
    bar_mesh_extract_->setFixedHeight(8);
    g_mesh->addWidget(new QLabel("Extraction:", this));
    g_mesh->addWidget(bar_mesh_extract_);
    root->addWidget(grp_mesh);

    // Export group
    auto* grp_export = new QGroupBox("Export", this);
    auto* g_export = new QVBoxLayout(grp_export);
    bar_export_ = new QProgressBar(this);
    bar_export_->setRange(0, 100);
    bar_export_->setValue(0);
    bar_export_->setFixedHeight(8);
    bar_export_->setTextVisible(false);
    g_export->addWidget(bar_export_);
    root->addWidget(grp_export);

    // State
    lbl_state_ = makeLabel("Idle");
    root->addWidget(lbl_state_);

    root->addStretch();
}

void MetricsPanel::update(const app::PipelineMetrics& m) {
    lbl_fps_capture_->setText(QString::number(m.capture_fps, 'f', 1) + " fps");
    lbl_fps_tracking_->setText(QString::number(m.tracking_fps, 'f', 1) + " fps");
    lbl_frame_count_->setText(QString::number(m.frame_count));
    lbl_backend_->setText(QString::fromStdString(m.backend));
    lbl_integrated_frames_->setText(QString::number(m.integrated_frames));
    lbl_volume_usage_->setText(QString::number(m.volume_usage_pct, 'f', 1) + " %");
    lbl_mesh_triangles_->setText(QString::number(m.mesh_triangles));
    lbl_icp_error_->setText(QString::number(m.icp_error, 'f', 4));

    const OverlapDisplay overlap = computeOverlapDisplay(m.icp_overlap_pct, m.icp_valid_model);
    lbl_icp_overlap_->setText(QString::fromStdString(overlap.text));
    // Re-polish only when the band actually changes; a text refresh alone (e.g.
    // "building… 1.1k" -> "building… 1.2k") never touches the style.
    if (overlap_band_.apply(overlap.band)) {
        lbl_icp_overlap_->setProperty("overlap", overlapBandProperty(overlap.band));
        lbl_icp_overlap_->style()->unpolish(lbl_icp_overlap_);
        lbl_icp_overlap_->style()->polish(lbl_icp_overlap_);
    }

    bar_volume_->setValue(static_cast<int>(m.volume_usage_pct));
    bar_mesh_extract_->setValue(static_cast<int>(m.mesh_extract_pct));
    bar_export_->setValue(static_cast<int>(m.export_pct));

    lbl_tracking_status_->setText(m.tracking_ok ? "OK" : "LOST");
    if (tracking_band_.apply(m.tracking_ok)) {
        lbl_tracking_status_->setProperty("status", m.tracking_ok ? "ok" : "lost");
        lbl_tracking_status_->style()->unpolish(lbl_tracking_status_);
        lbl_tracking_status_->style()->polish(lbl_tracking_status_);
    }

    QString state_str;
    switch (m.state) {
        case app::PipelineState::Idle:         state_str = "Idle";         break;
        case app::PipelineState::Running:      state_str = "Running";      break;
        case app::PipelineState::TrackingLost: state_str = "Tracking Lost";break;
        case app::PipelineState::Error:        state_str = "Error";        break;
        case app::PipelineState::Stopped:      state_str = "Stopped";      break;
    }
    lbl_state_->setText("State: " + state_str);
}

} // namespace gui
} // namespace kfusion
