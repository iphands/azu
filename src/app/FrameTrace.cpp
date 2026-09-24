#include "app/FrameTrace.h"

#include <Eigen/Geometry>

namespace kfusion {
namespace app {

namespace {
void writePose(std::ofstream& o, const Eigen::Matrix4f& T) {
    const Eigen::Quaternionf q(Eigen::Matrix3f(T.block<3,3>(0,0)));
    o << ',' << T(0,3) << ',' << T(1,3) << ',' << T(2,3)
      << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z();
}
} // namespace

bool FrameTrace::open(const std::string& path) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (open_) out_.close();
    out_.open(path, std::ios::out | std::ios::trunc);
    open_ = out_.is_open();
    if (open_) {
        out_.precision(7);
        out_ << "kind,frame_id,t_ms,quality,relocalizing,queued,"
                "tx,ty,tz,qw,qx,qy,qz,ptx,pty,ptz,pqw,pqx,pqy,pqz,"
                "inliers,valid_live,projected,valid_model,dist_filtered,angle_filtered,"
                "rms_m,final_step,converged,model_frame_id,outside_volume,degenerate_dofs,"
                "ms_preprocess,ms_icp,ms_track,ms_integrate,ms_raycast,tilt_err_deg,"
                "reloc_solves,reloc_refines,reloc_source,reloc_reject,reloc_consistent,"
                "reloc_violation,reloc_coverage,reloc_eig\n";
    }
    return open_;
}

void FrameTrace::write(const FrameTraceRow& r) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!open_) return;
    out_ << r.kind << ',' << r.frame_id << ',' << r.t_ms << ',' << r.quality << ','
         << int(r.relocalizing) << ',' << int(r.queued_for_integration);
    writePose(out_, r.pose);
    writePose(out_, r.predicted);
    out_ << ',' << r.inliers << ',' << r.valid_live << ',' << r.projected << ','
         << r.valid_model << ',' << r.dist_filtered << ',' << r.angle_filtered << ','
         << r.rms_m << ',' << r.final_step << ',' << int(r.converged) << ','
         << r.model_frame_id << ',' << r.outside_volume << ',' << r.degenerate_dofs << ','
         << r.ms_preprocess << ',' << r.ms_icp << ',' << r.ms_track << ','
         << r.ms_integrate << ',' << r.ms_raycast << ',' << r.tilt_err_deg << ','
         << r.reloc_solves << ',' << r.reloc_refines << ',' << r.reloc_source << ','
         << r.reloc_reject << ',' << r.reloc_consistent << ',' << r.reloc_violation << ','
         << r.reloc_coverage << ',' << r.reloc_eig << '\n';
    if (++unflushed_ >= 30) {
        out_.flush();
        unflushed_ = 0;
    }
}

void FrameTrace::close() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (open_) {
        out_.flush();
        out_.close();
    }
    open_ = false;
}

} // namespace app
} // namespace kfusion
