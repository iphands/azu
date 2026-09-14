#pragma once

#include <QWidget>
#include "app/FusionHyperparams.h"

QT_BEGIN_NAMESPACE
class QPushButton;
class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QLabel;
class QSlider;
class QGroupBox;
class QCheckBox;
QT_END_NAMESPACE

namespace kfusion {
namespace gui {

class NavigationGizmo;

class ControlPanel : public QWidget {
    Q_OBJECT
public:
    explicit ControlPanel(QWidget* parent = nullptr);

    app::FusionHyperparams hyperparamsFromUi() const;
    void setHyperparams(const app::FusionHyperparams& h);

signals:
    void startClicked();
    void stopClicked();
    void resetClicked();
    void exportPLYClicked();
    void exportGLBClicked();
    void modeChanged(int index); // 0=PointCloud, 1=Mesh
    void volumeCageToggled(bool visible);
    void threadsChanged(int n);
    void hyperparamsApplyClicked();
    void cameraRotationChanged(int pitch, int yaw, int roll);

public slots:
    void onPipelineStarted();
    void onPipelineStopped();
    void setExportEnabled(bool enabled);
    void setCameraRotation(int pitch, int yaw, int roll);
    void onPresetChanged(int index);

private:
    QPushButton* btn_start_  = nullptr;
    QPushButton* btn_stop_   = nullptr;
    QPushButton* btn_reset_  = nullptr;
    QPushButton* btn_ply_    = nullptr;
    QPushButton* btn_glb_    = nullptr;
    QComboBox*   combo_mode_ = nullptr;
    QCheckBox*   chk_volume_cage_ = nullptr;
    QSpinBox*    spin_threads_ = nullptr;
    QLabel*      lbl_status_ = nullptr;
    QComboBox*   combo_presets_ = nullptr;
    QPushButton* btn_toggle_hp_ = nullptr;
    QGroupBox*   grp_hp_ = nullptr;

    NavigationGizmo* nav_gizmo_ = nullptr;

    QDoubleSpinBox* spin_depth_min_  = nullptr;
    QDoubleSpinBox* spin_depth_max_  = nullptr;
    QDoubleSpinBox* spin_voxel_      = nullptr;
    QDoubleSpinBox* spin_trunc_      = nullptr;
    QDoubleSpinBox* spin_max_weight_ = nullptr;
    QSpinBox*       spin_resolution_ = nullptr;
    QDoubleSpinBox* spin_origin_x_   = nullptr;
    QDoubleSpinBox* spin_origin_y_   = nullptr;
    QDoubleSpinBox* spin_origin_z_   = nullptr;
    QDoubleSpinBox* spin_icp_dist_   = nullptr;
    QDoubleSpinBox* spin_icp_angle_  = nullptr;
    QSpinBox*       spin_icp_it0_    = nullptr;
    QSpinBox*       spin_icp_it1_    = nullptr;
    QSpinBox*       spin_icp_it2_    = nullptr;
    QPushButton*    btn_apply_hyper_ = nullptr;
    QComboBox*      combo_sr_scale_ = nullptr;

    void setupUI();
    void connectSignals();
};

} // namespace gui
} // namespace kfusion
