#pragma once

#include <QWidget>
#include <QLabel>
#include <QProgressBar>
#include "app/PipelineController.h"
#include "gui/FusionUiModel.h"

namespace kfusion {
namespace gui {

class MetricsPanel : public QWidget {
    Q_OBJECT
public:
    explicit MetricsPanel(QWidget* parent = nullptr);

    void update(const app::PipelineMetrics& m);

private:
    QLabel*       lbl_fps_capture_       = nullptr;
    QLabel*       lbl_fps_tracking_      = nullptr;
    QLabel*       lbl_frame_count_       = nullptr;
    QLabel*       lbl_tracking_status_   = nullptr;
    QLabel*       lbl_icp_error_         = nullptr;
    QLabel*       lbl_icp_overlap_       = nullptr;
    QLabel*       lbl_integrated_frames_ = nullptr;
    QLabel*       lbl_volume_usage_      = nullptr;
    QLabel*       lbl_mesh_triangles_    = nullptr;
    QLabel*       lbl_state_             = nullptr;
    QProgressBar* bar_volume_            = nullptr;
    QProgressBar* bar_mesh_extract_      = nullptr;
    QProgressBar* bar_export_            = nullptr;

    // Style bands are cached so a label re-polishes only on a band TRANSITION,
    // not on every 5 Hz metrics tick. Text still refreshes every tick; only the
    // (expensive) unpolish/polish is edge-triggered.
    BandCache<MetricsBand> overlap_band_;
    BandCache<bool>        tracking_band_;

    void setupUI();
    QLabel* makeLabel(const QString& text);
};

} // namespace gui
} // namespace kfusion
