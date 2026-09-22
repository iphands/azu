#include "gui/ControlPanel.h"
#include <QStyle>
#include <QVariant>
#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFrame>
#include <QScrollArea>
#include <QSlider>
#include "gui/NavigationGizmo.h"
#include "gui/FusionUiModel.h"
#include "utils/Logger.h"

#include <cmath>

namespace kfusion {
namespace gui {

namespace {

// Widget bounds only: setRange/setDecimals are per-control constraints, NOT a
// default table. Every value is seeded from FusionHyperparams::defaults() once
// via setHyperparams() at the end of setupUI(), so this TU never carries a
// second source of truth for what the tunables start at.
QDoubleSpinBox* makeDoubleSpin(double minV, double maxV, double step, int decimals, QWidget* parent) {
    auto* s = new QDoubleSpinBox(parent);
    s->setRange(minV, maxV);
    s->setSingleStep(step);
    s->setDecimals(decimals);
    s->setValue(minV); // transient floor; setHyperparams(defaults()) overwrites it
    s->setButtonSymbols(QAbstractSpinBox::PlusMinus);
    return s;
}

// Same (-180, 180] range the integer version had, without rounding the value.
float wrapDegrees(float deg) {
    float wrapped = std::fmod(deg, 360.0f);
    if (wrapped > 180.0f)       wrapped -= 360.0f;
    else if (wrapped < -180.0f) wrapped += 360.0f;
    return wrapped;
}

} // namespace

ControlPanel::ControlPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
    connectSignals();
}

void ControlPanel::setupUI() {
    auto* root = new QVBoxLayout(this);
    root->setSpacing(10);
    root->setContentsMargins(10, 10, 10, 10);

    lbl_status_ = new QLabel("Status: Idle", this);
    lbl_status_->setProperty("class", "StatusLabel");
    root->addWidget(lbl_status_);

    auto* line = new QFrame(this);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    root->addWidget(line);

    // 1. Capture Group
    auto* grp_capture = new QGroupBox("Capture", this);
    auto* v_cap = new QVBoxLayout(grp_capture);
    btn_start_ = new QPushButton("▶  Start Capture", this);
    btn_stop_  = new QPushButton("■  Stop Capture",  this);
    btn_reset_ = new QPushButton("↺  Reset Scan",    this);
    btn_start_->setProperty("class", "ActionBtn");
    btn_stop_ ->setProperty("class", "StopBtn");
    btn_stop_->setEnabled(false);
    v_cap->addWidget(btn_start_);
    v_cap->addWidget(btn_stop_);
    v_cap->addWidget(btn_reset_);
    root->addWidget(grp_capture);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* inner = new QWidget(scroll);
    auto* inner_layout = new QVBoxLayout(inner);
    inner_layout->setSpacing(10);
    inner_layout->setContentsMargins(0, 0, 6, 0);

    // 2. Viewport
    auto* grp_mode = new QGroupBox("Viewport", inner);
    auto* v_mode = new QVBoxLayout(grp_mode);
    combo_mode_ = new QComboBox(grp_mode);
    combo_mode_->addItem("Point Cloud");
    combo_mode_->addItem("Mesh");
    v_mode->addWidget(combo_mode_);
    chk_volume_cage_ = new QCheckBox("Capture volume", grp_mode);
    chk_volume_cage_->setToolTip(tr(
        "Shows the box your scan is kept inside. Nothing is reconstructed outside it.\n"
        "The box turns amber when the sensor leaves it — move back in, or enlarge it\n"
        "under Hyperparameters (size = resolution × voxel size)."));
    // Ordering load-bearing: setChecked before connectSignals() wires toggled keeps
    // checkbox/widget/renderer defaults (all true) in sync with no startup signal.
    chk_volume_cage_->setChecked(true);
    v_mode->addWidget(chk_volume_cage_);
    inner_layout->addWidget(grp_mode);

    // 3. Camera Controls
    auto* grp_cam = new QGroupBox("Camera Navigator", inner);
    auto* h_cam = new QVBoxLayout(grp_cam);
    
    nav_gizmo_ = new NavigationGizmo(grp_cam);
    h_cam->addWidget(nav_gizmo_, 0, Qt::AlignCenter);
    
    inner_layout->addWidget(grp_cam);

    // 4. Export
    auto* grp_export = new QGroupBox("Export", inner);
    auto* v_exp = new QVBoxLayout(grp_export);
    btn_ply_ = new QPushButton("Export PLY", grp_export);
    btn_glb_ = new QPushButton("Export GLB (Unity)", grp_export);
    btn_ply_->setProperty("class", "ActionBtn");
    btn_glb_->setProperty("class", "ActionBtn");
    btn_ply_->setEnabled(false);
    btn_glb_->setEnabled(false);
    v_exp->addWidget(btn_ply_);
    v_exp->addWidget(btn_glb_);
    inner_layout->addWidget(grp_export);

    // 5. Threads
    auto* thread_layout = new QHBoxLayout();
    thread_layout->addWidget(new QLabel("Threads (0=Auto):"));
    spin_threads_ = new QSpinBox(inner);
    spin_threads_->setRange(0, 256);
    spin_threads_->setValue(0);
    thread_layout->addWidget(spin_threads_);
    inner_layout->addLayout(thread_layout);

    // 6. Presets
    auto* grp_presets = new QGroupBox("Scanning Presets", inner);
    auto* v_pre = new QVBoxLayout(grp_presets);
    combo_presets_ = new QComboBox(grp_presets);
    combo_presets_->addItem("Custom / Current");
    combo_presets_->addItem("Helmet (Small, High Detail)");
    combo_presets_->addItem("Chair (Medium, Standard)");
    combo_presets_->addItem("Room (Large Environment)");
    combo_presets_->addItem("Human (Detail, Low Weight)");
    v_pre->addWidget(combo_presets_);
    inner_layout->addWidget(grp_presets);

    // 7. Toggle for Advanced
    btn_toggle_hp_ = new QPushButton("⚙  Show Advanced Configuration", inner);
    btn_toggle_hp_->setCheckable(true);
    inner_layout->addWidget(btn_toggle_hp_);

    // 8. Advanced Hyperparameters (Collapsed)
    grp_hp_ = new QGroupBox("Hyperparameters (Apply)", inner);
    auto* g = new QGridLayout(grp_hp_);
    int r = 0;

    g->addWidget(new QLabel("Depth min (m)", grp_hp_), r, 0);
    spin_depth_min_ = makeDoubleSpin(0.01, 3.0, 0.01, 3, grp_hp_);
    g->addWidget(spin_depth_min_, r++, 1);

    g->addWidget(new QLabel("Depth max (m)", grp_hp_), r, 0);
    // Upper bound is the device's usable range, so the widget cannot even offer a
    // value beyond what validateFusionHyperparams() would accept.
    spin_depth_max_ = makeDoubleSpin(0.2, kDeviceMaxDepthMeters, 0.1, 2, grp_hp_);
    g->addWidget(spin_depth_max_, r++, 1);

    g->addWidget(new QLabel("Voxel size (m)", grp_hp_), r, 0);
    spin_voxel_ = makeDoubleSpin(0.003, 0.05, 0.001, 3, grp_hp_);
    g->addWidget(spin_voxel_, r++, 1);

    g->addWidget(new QLabel("Truncation (m)", grp_hp_), r, 0);
    spin_trunc_ = makeDoubleSpin(0.01, 0.25, 0.005, 3, grp_hp_);
    g->addWidget(spin_trunc_, r++, 1);

    g->addWidget(new QLabel("Max weight", grp_hp_), r, 0);
    spin_max_weight_ = makeDoubleSpin(1.0, 512.0, 1.0, 0, grp_hp_);
    g->addWidget(spin_max_weight_, r++, 1);

    g->addWidget(new QLabel("Resolution (³)", grp_hp_), r, 0);
    spin_resolution_ = new QSpinBox(grp_hp_);
    spin_resolution_->setRange(64, 512);
    spin_resolution_->setSingleStep(32);
    g->addWidget(spin_resolution_, r++, 1);

    g->addWidget(new QLabel("Origin X", grp_hp_), r, 0);
    spin_origin_x_ = makeDoubleSpin(-4.0, 4.0, 0.05, 2, grp_hp_);
    g->addWidget(spin_origin_x_, r++, 1);
    g->addWidget(new QLabel("Origin Y", grp_hp_), r, 0);
    spin_origin_y_ = makeDoubleSpin(-4.0, 4.0, 0.05, 2, grp_hp_);
    g->addWidget(spin_origin_y_, r++, 1);
    g->addWidget(new QLabel("Origin Z", grp_hp_), r, 0);
    spin_origin_z_ = makeDoubleSpin(-2.0, 4.0, 0.05, 2, grp_hp_);
    g->addWidget(spin_origin_z_, r++, 1);

    g->addWidget(new QLabel("ICP dist (m)", grp_hp_), r, 0);
    spin_icp_dist_ = makeDoubleSpin(0.02, 0.5, 0.01, 3, grp_hp_);
    g->addWidget(spin_icp_dist_, r++, 1);

    g->addWidget(new QLabel("ICP angle (°)", grp_hp_), r, 0);
    spin_icp_angle_ = makeDoubleSpin(5.0, 90.0, 1.0, 0, grp_hp_);
    g->addWidget(spin_icp_angle_, r++, 1);

    g->addWidget(new QLabel("ICP iters coarse", grp_hp_), r, 0);
    spin_icp_it2_ = new QSpinBox(grp_hp_);
    spin_icp_it2_->setRange(1, 40);
    spin_icp_it2_->setValue(10);
    g->addWidget(spin_icp_it2_, r++, 1);

    g->addWidget(new QLabel("ICP iters mid", grp_hp_), r, 0);
    spin_icp_it1_ = new QSpinBox(grp_hp_);
    spin_icp_it1_->setRange(1, 40);
    spin_icp_it1_->setValue(5);
    g->addWidget(spin_icp_it1_, r++, 1);

    g->addWidget(new QLabel("ICP iters fine", grp_hp_), r, 0);
    spin_icp_it0_ = new QSpinBox(grp_hp_);
    spin_icp_it0_->setRange(1, 40);
    spin_icp_it0_->setValue(4);
    g->addWidget(spin_icp_it0_, r++, 1);

    g->addWidget(new QLabel("SR Scale", grp_hp_), r, 0);
    combo_sr_scale_ = new QComboBox(grp_hp_);
    combo_sr_scale_->addItem("2x");
    combo_sr_scale_->addItem("3x");
    combo_sr_scale_->addItem("4x");
    combo_sr_scale_->setCurrentIndex(0); // Default to 2x
    g->addWidget(combo_sr_scale_, r++, 1);

    btn_apply_hyper_ = new QPushButton("Apply hyperparameters", grp_hp_);
    btn_apply_hyper_->setProperty("class", "ActionBtn");
    g->addWidget(btn_apply_hyper_, r++, 0, 1, 2);

    grp_hp_->setVisible(false);
    inner_layout->addWidget(grp_hp_);

    inner_layout->addStretch();
    scroll->setWidget(inner);
    root->addWidget(scroll, 1);

    // Seed EVERY field from the one source of truth. setHyperparams() is the same
    // write path presets and the controller use, so the panel's starting state is
    // exactly FusionHyperparams::defaults() and cannot drift into a parallel table.
    setHyperparams(uiDefaultHyperparams());

    setMinimumWidth(260);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
}

void ControlPanel::connectSignals() {
    connect(btn_start_, &QPushButton::clicked, this, &ControlPanel::startClicked);
    connect(btn_stop_,  &QPushButton::clicked, this, &ControlPanel::stopClicked);
    connect(btn_reset_, &QPushButton::clicked, this, &ControlPanel::resetClicked);
    connect(btn_ply_,   &QPushButton::clicked, this, &ControlPanel::exportPLYClicked);
    connect(btn_glb_,   &QPushButton::clicked, this, &ControlPanel::exportGLBClicked);
    connect(combo_mode_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ControlPanel::modeChanged);
    connect(chk_volume_cage_, &QCheckBox::toggled,
            this, &ControlPanel::volumeCageToggled);
    connect(spin_threads_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ControlPanel::threadsChanged);
    connect(btn_apply_hyper_, &QPushButton::clicked, this, &ControlPanel::hyperparamsApplyClicked);

    connect(combo_presets_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ControlPanel::onPresetChanged);

    connect(btn_toggle_hp_, &QPushButton::toggled, this, [this](bool checked) {
        grp_hp_->setVisible(checked);
        btn_toggle_hp_->setText(checked ? "⚙  Hide Advanced Configuration" : "⚙  Show Advanced Configuration");
    });

    connect(nav_gizmo_, &NavigationGizmo::cameraRotationChanged, this, &ControlPanel::cameraRotationChanged);
}

app::FusionHyperparams ControlPanel::hyperparamsFromUi() const {
    app::FusionHyperparams h;
    h.min_depth = static_cast<float>(spin_depth_min_->value());
    h.max_depth = static_cast<float>(spin_depth_max_->value());
    h.sr_scale = combo_sr_scale_->currentText().toInt(); // Parse "2x", "3x", "4x"
    h.tsdf.voxel_size     = static_cast<float>(spin_voxel_->value());
    h.tsdf.truncation     = static_cast<float>(spin_trunc_->value());
    h.tsdf.max_weight     = static_cast<float>(spin_max_weight_->value());
    h.tsdf.resolution     = spin_resolution_->value();
    h.tsdf.origin         = Eigen::Vector3f(
        static_cast<float>(spin_origin_x_->value()),
        static_cast<float>(spin_origin_y_->value()),
        static_cast<float>(spin_origin_z_->value()));
    h.icp.dist_threshold  = static_cast<float>(spin_icp_dist_->value());
    h.icp.angle_threshold = static_cast<float>(spin_icp_angle_->value());
    h.icp.max_iterations[2] = spin_icp_it2_->value();
    h.icp.max_iterations[1] = spin_icp_it1_->value();
    h.icp.max_iterations[0] = spin_icp_it0_->value();
    return h;
}

void ControlPanel::setHyperparams(const app::FusionHyperparams& h) {
    spin_depth_min_->setValue(h.min_depth);
    spin_depth_max_->setValue(h.max_depth);
    // Set SR scale combo box
    QString scale_str = QString::number(h.sr_scale) + "x";
    int idx = combo_sr_scale_->findText(scale_str);
    if (idx >= 0) combo_sr_scale_->setCurrentIndex(idx);
    spin_voxel_->setValue(h.tsdf.voxel_size);
    spin_trunc_->setValue(h.tsdf.truncation);
    spin_max_weight_->setValue(h.tsdf.max_weight);
    spin_resolution_->setValue(h.tsdf.resolution);
    spin_origin_x_->setValue(h.tsdf.origin.x());
    spin_origin_y_->setValue(h.tsdf.origin.y());
    spin_origin_z_->setValue(h.tsdf.origin.z());
    spin_icp_dist_->setValue(h.icp.dist_threshold);
    spin_icp_angle_->setValue(h.icp.angle_threshold);
    spin_icp_it2_->setValue(h.icp.max_iterations[2]);
    spin_icp_it1_->setValue(h.icp.max_iterations[1]);
    spin_icp_it0_->setValue(h.icp.max_iterations[0]);
}

void ControlPanel::onPipelineStarted() {
    btn_start_->setEnabled(false);
    btn_stop_->setEnabled(true);
    lbl_status_->setText("Status: Running");
    lbl_status_->setProperty("status", "running");
    lbl_status_->style()->unpolish(lbl_status_);
    lbl_status_->style()->polish(lbl_status_);
}

void ControlPanel::onPipelineStopped() {
    btn_start_->setEnabled(true);
    btn_stop_->setEnabled(false);
    lbl_status_->setText("Status: Stopped");
    lbl_status_->setProperty("status", "stopped");
    lbl_status_->style()->unpolish(lbl_status_);
    lbl_status_->style()->polish(lbl_status_);
}

void ControlPanel::setExportEnabled(bool enabled) {
    btn_ply_->setEnabled(enabled);
    btn_glb_->setEnabled(enabled);
}

void ControlPanel::setBusy(bool busy) {
    // Latch lifecycle controls while a background pipeline operation (export
    // or reset) owns the controller. This never claims a pipeline state: the
    // caller restores the start/stop pair through onPipelineStarted()/
    // onPipelineStopped(), which re-derives it from the real controller.
    btn_start_->setEnabled(!busy);
    btn_stop_->setEnabled(false);
    btn_reset_->setEnabled(!busy);
}

void ControlPanel::setCameraRotation(float pitch, float yaw, float roll) {
    // Block signals to avoid infinite loop between mouse updates and slider updates
    nav_gizmo_->blockSignals(true);

    nav_gizmo_->setCameraRotation(wrapDegrees(pitch),
                                  wrapDegrees(yaw),
                                  wrapDegrees(roll));

    nav_gizmo_->blockSignals(false);
}

void ControlPanel::onPresetChanged(int index) {
    if (index == 0) return; // Custom / Current

    // Stage the whole preset as ONE value in the pure model, then apply it once.
    // Doing it piecemeal (mutating the current UI field by field) could leave a
    // half-written panel/controller if capture is running; one staged value plus
    // one setHyperparams() write plus one apply signal is atomic from the user's
    // point of view. MainWindow's apply handler re-validates the staged result.
    app::FusionHyperparams staged =
        applyFusionPreset(hyperparamsFromUi(), static_cast<FusionPreset>(index));

    setHyperparams(staged);
    KFLOGF_INFO("ControlPanel", "Preset applied: %s", combo_presets_->currentText().toStdString().c_str());
    emit hyperparamsApplyClicked();
}

} // namespace gui
} // namespace kfusion
