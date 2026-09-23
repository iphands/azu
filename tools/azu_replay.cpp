// azu_replay: replay a fakenect recording (fakenect-record <dir>) through the real
// pipeline, deterministically and as fast as the pipeline allows.
//
// Every depth/RGB sample goes through KinectSensor's real pairing
// (ingestDepth/ingestRgb, same code as live capture); each published frame is
// submitted to PipelineController and fully handled (graded, and if integrated
// also raycast) before the next one -- lockstep, no frame drops. So a run is
// repeatable and the numbers compare across code changes.
//
// Outputs in --out (default ./replay-out):
//   trace.csv    per-frame pipeline trace (include/app/FrameTrace.h)
//   frames.csv   per frame: quality, state, pose, accelerometer gravity,
//                cumulative yaw about gravity, tilt error vs gravity
//   mesh.ply     final mesh
//   summary.json the numbers below
// Summary: frames by grade, lost frames / first loss, total yaw tracked (about the
// gravity axis), tilt drift vs the accelerometer, and a loop-closure check (the
// last frame registered against a model fused from only the first frames).
//
// usage: azu_replay <recording_dir> [--out DIR] [--backend auto|cpu|cuda]
//                   [--volume front|centred] [--res N] [--voxel M]
//                   [--min-depth M] [--max-depth M] [--max-frames N] [--quiet]
#include "app/PipelineController.h"
#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "tracking/ICPTracker.h"
#include "tsdf/TSDFVolume.h"

#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

using namespace kfusion;
using Clock = std::chrono::steady_clock;

constexpr double kCountsPerG = 819.0;   // Kinect v1 accelerometer (libfreenect)
constexpr double kGravity    = 9.80665;

struct Options {
    std::string dir, out = "replay-out", backend = "auto", volume = "front";
    int res = 0, max_frames = 0;
    float voxel = 0.0f, min_depth = 0.0f, max_depth = 0.0f;
    bool quiet = false;
};

bool parseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--out") o.out = next("--out");
        else if (a == "--backend") o.backend = next("--backend");
        else if (a == "--volume") o.volume = next("--volume");
        else if (a == "--res") o.res = std::atoi(next("--res"));
        else if (a == "--voxel") o.voxel = std::strtof(next("--voxel"), nullptr);
        else if (a == "--min-depth") o.min_depth = std::strtof(next("--min-depth"), nullptr);
        else if (a == "--max-depth") o.max_depth = std::strtof(next("--max-depth"), nullptr);
        else if (a == "--max-frames") o.max_frames = std::atoi(next("--max-frames"));
        else if (a == "--quiet") o.quiet = true;
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return false;
        } else o.dir = a;
    }
    return !o.dir.empty();
}

// fakenect frame file: one header line, then raw bytes.
bool readPayload(const std::string& path, size_t bytes, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string header;
    std::getline(f, header);
    out.resize(bytes);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(f.gcount()) == bytes;
}

// a-*.dump is a raw freenect_raw_tilt_state; its first three int16 are the
// accelerometer counts.
bool readAccel(const std::string& path, Eigen::Vector3d& g) {
    std::ifstream f(path, std::ios::binary);
    int16_t xyz[3];
    if (!f.read(reinterpret_cast<char*>(xyz), sizeof(xyz))) return false;
    g = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]) * (kGravity / kCountsPerG);
    return true;
}

double angleDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const double c = a.normalized().dot(b.normalized());
    return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

struct FrameRecord {
    uint64_t id = 0;
    int quality = -1;
    int state = 0;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
    bool has_accel = false;
};

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        std::fprintf(stderr,
                     "usage: azu_replay <recording_dir> [--out DIR] [--backend auto|cpu|cuda]\n"
                     "                  [--volume front|centred] [--res N] [--voxel M]\n"
                     "                  [--min-depth M] [--max-depth M] [--max-frames N] [--quiet]\n");
        return 2;
    }
    mkdir(opt.out.c_str(), 0755);

    std::ifstream index(opt.dir + "/INDEX.txt");
    if (!index) {
        std::fprintf(stderr, "no INDEX.txt in %s\n", opt.dir.c_str());
        return 1;
    }

    // ---- pipeline
    const sensor::PreprocessBackend backend = sensor::parseBackendName(opt.backend);
    app::PipelineController pc(backend);
    app::FusionHyperparams hp = pc.hyperparamsSnapshot();
    if (opt.min_depth > 0.0f) hp.min_depth = opt.min_depth;
    if (opt.max_depth > 0.0f) hp.max_depth = opt.max_depth;
    if (opt.res > 0) hp.tsdf.resolution = opt.res;
    if (opt.voxel > 0.0f) hp.tsdf.voxel_size = opt.voxel;
    const float extent = hp.tsdf.voxel_size * static_cast<float>(hp.tsdf.resolution);
    if (opt.volume == "centred") {
        hp.tsdf.origin = Eigen::Vector3f::Constant(-0.5f * extent);
    } else {
        hp.tsdf.origin = Eigen::Vector3f(-0.5f * extent, -0.5f * extent, 0.0f);
    }
    pc.setHyperparams(hp);
    pc.setTracePath(opt.out + "/trace.csv");
    if (!pc.startOffline()) {
        std::fprintf(stderr, "pipeline failed to start\n");
        return 1;
    }
    const std::string backend_label = pc.metricsSnapshot().backend;
    std::printf("replay %s: backend %s, volume %s %d^3 x %.3f m (%.2f m), depth %.2f-%.2f m\n",
                opt.dir.c_str(), backend_label.c_str(), opt.volume.c_str(), hp.tsdf.resolution,
                hp.tsdf.voxel_size, extent, hp.min_depth, hp.max_depth);

    // ---- pairing through the real sensor code
    sensor::KinectSensor pairing;
    std::deque<std::shared_ptr<sensor::RawFrame>> published;
    pairing.setFrameCallback([&](std::shared_ptr<sensor::RawFrame> f) { published.push_back(std::move(f)); });

    std::vector<FrameRecord> records;
    std::vector<std::vector<uint16_t>> first_depth;   // raw depth of the first frames (loop check)
    std::vector<Eigen::Matrix4f> first_pose;
    std::vector<uint16_t> last_depth;
    constexpr size_t kLoopModelFrames = 30;

    Eigen::Vector3d accel_now = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_sum = Eigen::Vector3d::Zero();
    int accel_count = 0;
    bool have_accel = false;
    std::vector<uint8_t> buf;
    const size_t depth_bytes = static_cast<size_t>(sensor::DEPTH_WIDTH) * sensor::DEPTH_HEIGHT * 2;
    const size_t rgb_bytes = static_cast<size_t>(sensor::RGB_WIDTH) * sensor::RGB_HEIGHT * 3;
    const auto t_start = Clock::now();
    int timeouts = 0;
    std::string line;
    while (std::getline(index, line)) {
        if (line.size() < 3) continue;
        char type = 0;
        double host_time = 0.0;
        unsigned int ticks = 0;
        if (std::sscanf(line.c_str(), "%c-%lf-%u-", &type, &host_time, &ticks) != 3) continue;
        const std::string path = opt.dir + "/" + line;
        if (type == 'a') {
            // ~10 samples per frame; the mean over the frame interval removes
            // most hand jitter from the gravity estimate.
            Eigen::Vector3d g;
            if (readAccel(path, g)) {
                accel_sum += g;
                ++accel_count;
            }
            continue;
        }
        if (type == 'd' && accel_count > 0) {
            accel_now = accel_sum / accel_count;
            have_accel = true;
            accel_sum.setZero();
            accel_count = 0;
        }
        if (type == 'd') {
            if (!readPayload(path, depth_bytes, buf)) continue;
            pairing.ingestDepth(buf.data(), ticks);
        } else if (type == 'r') {
            if (!readPayload(path, rgb_bytes, buf)) continue;
            pairing.ingestRgb(buf.data(), ticks);
        }

        while (!published.empty()) {
            std::shared_ptr<sensor::RawFrame> f = std::move(published.front());
            published.pop_front();
            const uint64_t id = f->frame_id;
            std::vector<uint16_t> depth_copy(f->depth);
            pc.submitRawFrame(std::move(f));
            if (!pc.waitIdle(std::chrono::seconds(20))) ++timeouts;

            const app::PipelineMetrics m = pc.metricsSnapshot();
            FrameRecord r;
            r.id = id;
            r.quality = records.empty() ? -1 : m.tracking_quality;
            r.state = static_cast<int>(m.state);
            r.pose = pc.currentPose();
            r.accel = accel_now;
            r.has_accel = have_accel;
            records.push_back(r);
            if (first_depth.size() < kLoopModelFrames) {
                first_depth.push_back(depth_copy);
                first_pose.push_back(r.pose);
            }
            last_depth = std::move(depth_copy);

            if (!opt.quiet && records.size() % 50 == 0) {
                std::printf("  frame %zu: quality %d state %d t=(%.3f %.3f %.3f)\n", records.size(),
                            r.quality, r.state, r.pose(0, 3), r.pose(1, 3), r.pose(2, 3));
                std::fflush(stdout);
            }
        }
        if (opt.max_frames > 0 && records.size() >= static_cast<size_t>(opt.max_frames)) break;
    }
    const double wall_s = std::chrono::duration<double>(Clock::now() - t_start).count();
    const bool mesh_ok = pc.exportPLY(opt.out + "/mesh.ply");
    pc.stop();

    if (records.size() < 2) {
        std::fprintf(stderr, "recording produced %zu frames\n", records.size());
        return 1;
    }

    // ---- per-frame analysis
    // World = first camera. "Up" is the first accelerometer reading (gravity
    // reaction) in camera axes; the accelerometer axes are taken as the camera
    // axes up to per-axis sign, chosen as the flip that best explains the run.
    // Gravity reference: the first frame that has an accelerometer mean (the
    // first depth frame can arrive before any sample), paired with that
    // frame's pose.
    Eigen::Vector3d a0(0, -1, 0);
    Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
    for (const FrameRecord& r : records) {
        if (r.has_accel) {
            a0 = r.accel;
            R0 = r.pose.block<3,3>(0,0).cast<double>();
            break;
        }
    }
    Eigen::Vector3d best_sign(1, 1, 1);
    double best_err = 1e9;
    for (int m = 0; m < 8; ++m) {
        const Eigen::Vector3d sgn((m & 1) ? -1 : 1, (m & 2) ? -1 : 1, (m & 4) ? -1 : 1);
        const Eigen::Vector3d g_world = R0 * sgn.cwiseProduct(a0);
        double sum = 0.0;
        int n = 0;
        for (const FrameRecord& r : records) {
            if (!r.has_accel || r.quality == 2) continue;
            const Eigen::Vector3d pred = r.pose.block<3,3>(0,0).cast<double>().transpose() * g_world;
            sum += angleDeg(sgn.cwiseProduct(r.accel), pred);
            ++n;
        }
        if (n > 0 && sum / n < best_err) {
            best_err = sum / n;
            best_sign = sgn;
        }
    }
    const Eigen::Vector3d g_world = R0 * best_sign.cwiseProduct(a0);
    const Eigen::Vector3d up = g_world.normalized();

    int good = 0, poor = 0, failed = 0, lost = 0, first_lost = -1;
    double yaw_total = 0.0, tilt_max = 0.0, tilt_sum = 0.0;
    int tilt_n = 0;
    std::ofstream fcsv(opt.out + "/frames.csv");
    fcsv << "idx,frame_id,quality,state,tx,ty,tz,qw,qx,qy,qz,ax,ay,az,yaw_total_deg,tilt_err_deg\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const FrameRecord& r = records[i];
        if (r.quality == 0) ++good;
        else if (r.quality == 1) ++poor;
        else if (r.quality == 2) ++failed;
        if (r.state == static_cast<int>(app::PipelineState::TrackingLost)) {
            ++lost;
            if (first_lost < 0) first_lost = static_cast<int>(i);
        }
        if (i > 0) {
            const Eigen::Matrix3f dR =
                records[i - 1].pose.block<3,3>(0,0).transpose() * r.pose.block<3,3>(0,0);
            // Body-frame increment -> world frame axis before projecting on up.
            const Eigen::AngleAxisf aa(dR);
            const Eigen::Vector3d axis_world =
                (records[i - 1].pose.block<3,3>(0,0) * aa.axis()).cast<double>();
            yaw_total += aa.angle() * axis_world.dot(up);
        }
        double tilt = 0.0;
        if (r.has_accel) {
            const Eigen::Vector3d pred = r.pose.block<3,3>(0,0).cast<double>().transpose() * g_world;
            tilt = angleDeg(best_sign.cwiseProduct(r.accel), pred);
            tilt_max = std::max(tilt_max, tilt);
            tilt_sum += tilt;
            ++tilt_n;
        }
        const Eigen::Quaternionf q(Eigen::Matrix3f(r.pose.block<3,3>(0,0)));
        fcsv << i << ',' << r.id << ',' << r.quality << ',' << r.state << ',' << r.pose(0, 3) << ','
             << r.pose(1, 3) << ',' << r.pose(2, 3) << ',' << q.w() << ',' << q.x() << ',' << q.y()
             << ',' << q.z() << ',' << r.accel.x() << ',' << r.accel.y() << ',' << r.accel.z() << ','
             << yaw_total * 180.0 / M_PI << ',' << tilt << '\n';
    }

    // ---- loop closure: last frame vs a model fused from the first frames only
    // (independent CPU fusion with the poses tracked for those frames).
    tsdf::TSDFParams lp = hp.tsdf;
    tsdf::TSDFVolume loop_model(lp);
    sensor::FrameData fd;
    const std::vector<uint8_t> no_rgb(rgb_bytes, 0);
    for (size_t i = 0; i < first_depth.size(); ++i) {
        sensor::buildFrameData(first_depth[i].data(), no_rgb.data(), fd, hp.min_depth, hp.max_depth);
        loop_model.integrate(fd.depth_meters.data(), nullptr, first_pose[i], sensor::FX, sensor::FY,
                             sensor::CX, sensor::CY, fd.width, fd.height, hp.min_depth, hp.max_depth);
    }
    const Eigen::Matrix4f end_pose = records.back().pose;
    tracking::ModelFrame model;
    loop_model.raycast(end_pose, sensor::FX, sensor::FY, sensor::CX, sensor::CY, sensor::FRAME_W,
                       sensor::FRAME_H, model.vertices.data(), model.normals.data(), model.colors.data());
    model.pose = end_pose;
    sensor::buildFrameData(last_depth.data(), no_rgb.data(), fd, hp.min_depth, hp.max_depth);
    sensor::computeNormals(fd);
    sensor::FramePyramid pyr;
    sensor::buildFramePyramid(fd, pyr);
    tracking::ICPTracker icp;
    const tracking::ICPResult loop = icp.track(pyr, model, end_pose, model.pose);
    const Eigen::Matrix4f corr = end_pose.inverse() * loop.pose;
    const double loop_t = corr.block<3,1>(0,3).norm();
    const double loop_r = Eigen::AngleAxisd(corr.block<3,3>(0,0).cast<double>()).angle() * 180.0 / M_PI;
    const bool loop_valid = loop.inliers > 5000 && loop.pose.allFinite();

    const double yaw_deg = yaw_total * 180.0 / M_PI;
    std::printf("\nframes %zu (%.1f s, %.1f fps): good %d, poor %d, failed %d, lost frames %d "
                "(first at frame %d), timeouts %d\n",
                records.size(), wall_s, records.size() / wall_s, good, poor, failed, lost, first_lost,
                timeouts);
    std::printf("yaw tracked about gravity: %.1f deg\n", yaw_deg);
    std::printf("tilt vs accelerometer: mean %.2f deg, max %.2f deg (axis signs %+.0f %+.0f %+.0f)\n",
                tilt_n ? tilt_sum / tilt_n : 0.0, tilt_max, best_sign.x(), best_sign.y(), best_sign.z());
    std::printf("loop closure (last frame vs first-%zu-frame model): %s %.1f mm / %.2f deg "
                "(inliers %d)\n",
                first_depth.size(), loop_valid ? "" : "[UNRELIABLE]", loop_t * 1e3, loop_r, loop.inliers);
    std::printf("outputs: %s/{trace.csv,frames.csv,summary.json%s}\n", opt.out.c_str(),
                mesh_ok ? ",mesh.ply" : "");

    std::ofstream js(opt.out + "/summary.json");
    js << "{\n"
       << "  \"recording\": \"" << opt.dir << "\",\n"
       << "  \"backend\": \"" << backend_label << "\",\n"
       << "  \"volume\": \"" << opt.volume << "\", \"resolution\": " << hp.tsdf.resolution
       << ", \"voxel_size\": " << hp.tsdf.voxel_size << ",\n"
       << "  \"frames\": " << records.size() << ", \"good\": " << good << ", \"poor\": " << poor
       << ", \"failed\": " << failed << ", \"lost_frames\": " << lost << ", \"first_lost\": " << first_lost
       << ", \"timeouts\": " << timeouts << ",\n"
       << "  \"wall_seconds\": " << wall_s << ",\n"
       << "  \"yaw_tracked_deg\": " << yaw_deg << ",\n"
       << "  \"tilt_mean_deg\": " << (tilt_n ? tilt_sum / tilt_n : 0.0) << ", \"tilt_max_deg\": " << tilt_max
       << ",\n"
       << "  \"loop_trans_mm\": " << loop_t * 1e3 << ", \"loop_rot_deg\": " << loop_r
       << ", \"loop_inliers\": " << loop.inliers << ", \"loop_reliable\": " << (loop_valid ? "true" : "false")
       << "\n}\n";
    return 0;
}
