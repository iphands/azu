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
// gravity axis), tilt drift vs the accelerometer, a loop-closure check (the
// last frame registered against a model fused from only the first frames), and
// one row per loss episode (when lost, when and from which candidate
// re-acquired, how the next 30 frames went, anything integrated during
// probation, error vs --reference).
//
// Relocalization experiments (ranges are recording depth frames, 0-based in
// INDEX order, half-open A:B; frames.csv's depth_index column):
//   --kidnap A:B[,C:D]  drop those frames: the camera jumps from A-1 to B
//   --blank A:B[,C:D]   zero their depth (the accelerometer keeps running):
//                       tracking is lost after 3 and must re-acquire after B
//   --reference DIR     a replay of the same recording without them: every
//                       frame is compared with that run's pose for the same
//                       depth frame (frames.csv ref_err_*), when it was Good
//
// usage: azu_replay <recording_dir> [--out DIR] [--backend auto|cpu|cuda]
//                   [--preset room|helmet|chair|human|none] [--intrinsics device|legacy]
//                   [--volume front|centred] [--res N] [--voxel M]
//                   [--min-depth M] [--max-depth M] [--max-frames N] [--quiet]
//                   [--kidnap A:B[,..]] [--blank A:B[,..]] [--reference DIR]
// The preset defaults to room (recordings are turns in place); `none` keeps
// the pipeline defaults with a front-anchored 256^3 x 1 cm object volume.
#include "FakenectRecording.h"
#include "app/PipelineController.h"
#include "gui/FusionUiModel.h"
#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "tracking/ICPTracker.h"
#include "tracking/TrackingPolicy.h"
#include "tsdf/TSDFVolume.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace {

using namespace kfusion;
using Clock = std::chrono::steady_clock;

constexpr double kGravity    = 9.80665;

struct Options {
    std::string dir, out = "replay-out", backend = "auto", volume = "front", preset = "room";
    std::string intrinsics = "device";
    int res = 0, max_frames = 0;
    float voxel = 0.0f, min_depth = 0.0f, max_depth = 0.0f;
    bool quiet = false;
    bool volume_set = false;
    std::vector<std::pair<int, int>> kidnap, blank;   // recording depth frames [A, B)
    std::string reference;
};

// "A:B[,C:D...]" with 0 <= A < B.
bool parseRanges(const std::string& text, std::vector<std::pair<int, int>>& out) {
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const size_t colon = item.find(':');
        if (colon == std::string::npos) return false;
        const int a = std::atoi(item.substr(0, colon).c_str());
        const int b = std::atoi(item.substr(colon + 1).c_str());
        if (a < 0 || b <= a) return false;
        out.push_back({a, b});
    }
    return !out.empty();
}

bool inRanges(int i, const std::vector<std::pair<int, int>>& ranges) {
    for (const auto& r : ranges) {
        if (i >= r.first && i < r.second) return true;
    }
    return false;
}

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
        else if (a == "--volume") { o.volume = next("--volume"); o.volume_set = true; }
        else if (a == "--preset") o.preset = next("--preset");
        else if (a == "--intrinsics") o.intrinsics = next("--intrinsics");
        else if (a == "--res") o.res = std::atoi(next("--res"));
        else if (a == "--voxel") o.voxel = std::strtof(next("--voxel"), nullptr);
        else if (a == "--min-depth") o.min_depth = std::strtof(next("--min-depth"), nullptr);
        else if (a == "--max-depth") o.max_depth = std::strtof(next("--max-depth"), nullptr);
        else if (a == "--max-frames") o.max_frames = std::atoi(next("--max-frames"));
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--kidnap" || a == "--blank") {
            if (!parseRanges(next(a.c_str()), a == "--kidnap" ? o.kidnap : o.blank)) {
                std::fprintf(stderr, "%s wants A:B[,C:D...] with 0 <= A < B\n", a.c_str());
                return false;
            }
        }
        else if (a == "--reference") o.reference = next("--reference");
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return false;
        } else o.dir = a;
    }
    return !o.dir.empty();
}

// A number field from fakenect-record's device.json (flat enough for a scan).
bool jsonNumber(const std::string& text, const std::string& key, double* out) {
    const size_t k = text.find("\"" + key + "\"");
    if (k == std::string::npos) return false;
    const size_t colon = text.find(':', k);
    if (colon == std::string::npos) return false;
    char* end = nullptr;
    const double v = std::strtod(text.c_str() + colon + 1, &end);
    if (end == text.c_str() + colon + 1) return false;
    *out = v;
    return true;
}

double angleDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const double c = a.normalized().dot(b.normalized());
    return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

struct FrameRecord {
    uint64_t id = 0;
    int depth_index = -1;   // the recording's depth frame (INDEX order)
    int quality = -1;
    int state = 0;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
    bool has_accel = false;
};

// A CSV with a header row, as rows of named fields (no quoting: our files).
struct Csv {
    std::map<std::string, int> col;
    std::vector<std::vector<std::string>> rows;
    const std::string& at(size_t r, const char* name) const {
        static const std::string none;
        const auto it = col.find(name);
        if (it == col.end() || it->second >= static_cast<int>(rows[r].size())) return none;
        return rows[r][it->second];
    }
};

bool readCsv(const std::string& path, Csv& out) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line)) return false;
    auto split = [](const std::string& l) {
        std::vector<std::string> f;
        std::stringstream ss(l);
        std::string x;
        while (std::getline(ss, x, ',')) f.push_back(x);
        return f;
    };
    const std::vector<std::string> head = split(line);
    for (size_t i = 0; i < head.size(); ++i) out.col[head[i]] = static_cast<int>(i);
    while (std::getline(in, line)) out.rows.push_back(split(line));
    return true;
}

Eigen::Matrix4f poseFromCsv(const Csv& c, size_t r) {
    auto f = [&](const char* n) { return std::strtof(c.at(r, n).c_str(), nullptr); };
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3,3>(0,0) = Eigen::Quaternionf(f("qw"), f("qx"), f("qy"), f("qz")).normalized().toRotationMatrix();
    m.block<3,1>(0,3) = Eigen::Vector3f(f("tx"), f("ty"), f("tz"));
    return m;
}

struct PoseErr {
    double mm = -1.0, deg = -1.0;
};

PoseErr poseErr(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b) {
    PoseErr e;
    e.mm = (a.block<3,1>(0,3) - b.block<3,1>(0,3)).norm() * 1e3;
    const Eigen::Matrix3d r = (a.block<3,3>(0,0).transpose() * b.block<3,3>(0,0)).cast<double>();
    e.deg = Eigen::AngleAxisd(r).angle() * 180.0 / M_PI;
    return e;
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        std::fprintf(stderr,
                     "usage: azu_replay <recording_dir> [--out DIR] [--backend auto|cpu|cuda]\n"
                     "                  [--preset room|helmet|chair|human|none] [--intrinsics device|legacy]\n"
                     "                  [--volume front|centred] [--res N] [--voxel M]\n"
                     "                  [--min-depth M] [--max-depth M] [--max-frames N] [--quiet]\n"
                     "                  [--kidnap A:B[,..]] [--blank A:B[,..]] [--reference DIR]\n");
        return 2;
    }
    // The reference run's Good poses by recording depth frame.
    std::unordered_map<int, Eigen::Matrix4f> reference;
    if (!opt.reference.empty()) {
        Csv ref;
        if (!readCsv(opt.reference + "/frames.csv", ref) || !ref.col.count("depth_index")) {
            std::fprintf(stderr, "--reference: no frames.csv with a depth_index column in %s\n",
                         opt.reference.c_str());
            return 2;
        }
        for (size_t r = 0; r < ref.rows.size(); ++r) {
            if (ref.at(r, "quality") != "0") continue;
            reference[std::atoi(ref.at(r, "depth_index").c_str())] = poseFromCsv(ref, r);
        }
    }
    mkdir(opt.out.c_str(), 0755);

    bool index_ok = false;
    const std::vector<azu_rec::Entry> entries = azu_rec::parseIndex(opt.dir, &index_ok);
    if (!index_ok) {
        std::fprintf(stderr, "no INDEX.txt in %s\n", opt.dir.c_str());
        return 1;
    }

    // ---- pipeline
    const sensor::PreprocessBackend backend = sensor::parseBackendName(opt.backend);
    app::PipelineController pc(backend);
    app::FusionHyperparams hp = pc.hyperparamsSnapshot();
    // --preset applies the GUI preset (volume placement included); --volume,
    // --res, --voxel and the depth band then override it only when given.
    bool placed_by_preset = false;
    if (opt.preset != "none") {
        const std::pair<const char*, gui::FusionPreset> names[] = {
            {"helmet", gui::FusionPreset::kHelmet}, {"chair", gui::FusionPreset::kChair},
            {"room", gui::FusionPreset::kRoom}, {"human", gui::FusionPreset::kHuman}};
        bool found = false;
        for (const auto& n : names) {
            if (opt.preset == n.first) {
                hp = gui::applyFusionPreset(hp, n.second);
                found = placed_by_preset = true;
            }
        }
        if (!found) {
            std::fprintf(stderr, "unknown --preset %s (room|helmet|chair|human|none)\n", opt.preset.c_str());
            return 2;
        }
    }
    if (opt.min_depth > 0.0f) hp.min_depth = opt.min_depth;
    if (opt.max_depth > 0.0f) hp.max_depth = opt.max_depth;
    if (opt.res > 0) hp.tsdf.resolution = opt.res;
    if (opt.voxel > 0.0f) hp.tsdf.voxel_size = opt.voxel;
    const float extent = hp.tsdf.voxel_size * static_cast<float>(hp.tsdf.resolution);
    const bool volume_given = opt.volume_set || opt.res > 0 || opt.voxel > 0.0f;
    if (!placed_by_preset || volume_given) {
        if (opt.volume == "centred") {
            hp.tsdf.origin = Eigen::Vector3f::Constant(-0.5f * extent);
        } else {
            hp.tsdf.origin = Eigen::Vector3f(-0.5f * extent, -0.5f * extent, 0.0f);
        }
    }
    if (placed_by_preset && !volume_given) {
        opt.volume = hp.tsdf.origin.z() < 0.0f ? "centred" : "front";
    }
    pc.setHyperparams(hp);
    pc.setTracePath(opt.out + "/trace.csv");

    // Depth intrinsics: the recording's device.json (the unit's factory
    // registration) unless --intrinsics legacy.
    sensor::CameraIntrinsics K = sensor::kLegacyIntrinsics;
    std::string intr_source = "legacy";
    if (opt.intrinsics == "device") {
        std::ifstream dj(opt.dir + "/device.json");
        const std::string text((std::istreambuf_iterator<char>(dj)), std::istreambuf_iterator<char>());
        double ref_dist = 0.0, ref_px = 0.0;
        if (jsonNumber(text, "reference_distance", &ref_dist) &&
            jsonNumber(text, "reference_pixel_size", &ref_px) &&
            sensor::intrinsicsFromZeroPlane(ref_dist, ref_px, &K)) {
            intr_source = "device.json";
        }
    } else if (opt.intrinsics != "legacy") {
        std::fprintf(stderr, "--intrinsics must be device or legacy\n");
        return 2;
    }
    pc.setIntrinsics(K);
    if (!pc.startOffline()) {
        std::fprintf(stderr, "pipeline failed to start\n");
        return 1;
    }
    const std::string backend_label = pc.metricsSnapshot().backend;
    std::printf("replay %s: backend %s, preset %s, volume %s %d^3 x %.3f m (%.2f m), depth %.2f-%.2f m, "
                "fx %.1f (%s)\n",
                opt.dir.c_str(), backend_label.c_str(), opt.preset.c_str(), opt.volume.c_str(), hp.tsdf.resolution,
                hp.tsdf.voxel_size, extent, hp.min_depth, hp.max_depth, K.fx, intr_source.c_str());

    // ---- pairing through the real sensor code
    sensor::KinectSensor pairing;
    std::deque<std::shared_ptr<sensor::RawFrame>> published;
    pairing.setFrameCallback([&](std::shared_ptr<sensor::RawFrame> f) { published.push_back(std::move(f)); });

    std::vector<FrameRecord> records;
    std::vector<std::vector<uint16_t>> first_depth;   // raw depth of the first frames (loop check)
    std::vector<Eigen::Matrix4f> first_pose;
    std::vector<uint16_t> last_depth;
    constexpr size_t kLoopModelFrames = 30;

    std::vector<uint8_t> buf;
    const size_t depth_bytes = static_cast<size_t>(sensor::DEPTH_WIDTH) * sensor::DEPTH_HEIGHT * 2;
    const size_t rgb_bytes = static_cast<size_t>(sensor::RGB_WIDTH) * sensor::RGB_HEIGHT * 3;
    const auto t_start = Clock::now();
    int timeouts = 0;
    int depth_index = -1;
    std::unordered_map<uint32_t, int> depth_index_of;   // ticks -> index, until published
    for (const azu_rec::Entry& e : entries) {
        const char type = e.type;
        const uint32_t ticks = e.ticks;
        const std::string path = opt.dir + "/" + e.file;
        if (type == 'a') {
            // ~10 samples per frame. The sensor averages them over each frame
            // interval and hands the mean to the pipeline with the frame (the
            // gravity gate), exactly as for a live device.
            int16_t xyz[3];
            if (azu_rec::readAccelCounts(opt.dir, e, xyz)) {
                const double k = kGravity / azu_rec::kCountsPerG;
                pairing.ingestAccel(xyz[0] * k, xyz[1] * k, xyz[2] * k);
            }
            continue;
        }
        if (type == 'd') {
            ++depth_index;
            if (inRanges(depth_index, opt.kidnap)) continue;   // never happened: the camera jumps
            if (!azu_rec::readPayload(path, depth_bytes, buf)) continue;
            if (inRanges(depth_index, opt.blank)) std::fill(buf.begin(), buf.end(), 0);   // raw 0 = no depth
            depth_index_of[ticks] = depth_index;
            pairing.ingestDepth(buf.data(), ticks);
        } else if (type == 'r') {
            if (!azu_rec::readPayload(path, rgb_bytes, buf)) continue;
            pairing.ingestRgb(buf.data(), ticks);
        }

        while (!published.empty()) {
            std::shared_ptr<sensor::RawFrame> f = std::move(published.front());
            published.pop_front();
            const uint64_t id = f->frame_id;
            int frame_depth_index = -1;
            if (const auto di = depth_index_of.find(f->depth_ticks); di != depth_index_of.end()) {
                frame_depth_index = di->second;
                depth_index_of.erase(di);   // ticks wrap every ~71.6 s
            }
            const Eigen::Vector3d accel(f->accel[0], f->accel[1], f->accel[2]);
            const bool has_accel = f->accel_valid;
            std::vector<uint16_t> depth_copy(f->depth);
            pc.submitRawFrame(std::move(f));
            if (!pc.waitIdle(std::chrono::seconds(20))) ++timeouts;

            const app::PipelineMetrics m = pc.metricsSnapshot();
            FrameRecord r;
            r.id = id;
            r.depth_index = frame_depth_index;
            r.quality = records.empty() ? -1 : m.tracking_quality;
            r.state = static_cast<int>(m.state);
            r.pose = pc.currentPose();
            r.accel = accel;
            r.has_accel = has_accel;
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
    std::vector<double> tilt_of(records.size(), -1.0);
    std::vector<PoseErr> ref_err(records.size());
    std::ofstream fcsv(opt.out + "/frames.csv");
    fcsv << "idx,frame_id,quality,state,tx,ty,tz,qw,qx,qy,qz,ax,ay,az,yaw_total_deg,tilt_err_deg,"
            "depth_index,ref_err_mm,ref_err_deg\n";
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
            tilt_of[i] = tilt;
        }
        if (const auto ref = reference.find(r.depth_index); ref != reference.end()) {
            ref_err[i] = poseErr(r.pose, ref->second);
        }
        const Eigen::Quaternionf q(Eigen::Matrix3f(r.pose.block<3,3>(0,0)));
        fcsv << i << ',' << r.id << ',' << r.quality << ',' << r.state << ',' << r.pose(0, 3) << ','
             << r.pose(1, 3) << ',' << r.pose(2, 3) << ',' << q.w() << ',' << q.x() << ',' << q.y()
             << ',' << q.z() << ',' << r.accel.x() << ',' << r.accel.y() << ',' << r.accel.z() << ','
             << yaw_total * 180.0 / M_PI << ',' << tilt << ',' << r.depth_index << ',' << ref_err[i].mm << ','
             << ref_err[i].deg << '\n';
    }

    // ---- loop closure: last frame vs a model fused from the first frames only
    // (independent CPU fusion with the poses tracked for those frames).
    tsdf::TSDFParams lp = hp.tsdf;
    tsdf::TSDFVolume loop_model(lp);
    sensor::FrameData fd;
    const std::vector<uint8_t> no_rgb(rgb_bytes, 0);
    for (size_t i = 0; i < first_depth.size(); ++i) {
        sensor::buildFrameData(first_depth[i].data(), no_rgb.data(), fd, hp.min_depth, hp.max_depth, K);
        loop_model.integrate(fd.depth_meters.data(), nullptr, first_pose[i], K.fx, K.fy, K.cx, K.cy,
                             fd.width, fd.height, hp.min_depth, hp.max_depth);
    }
    const Eigen::Matrix4f end_pose = records.back().pose;
    tracking::ModelFrame model;
    loop_model.raycast(end_pose, K.fx, K.fy, K.cx, K.cy, sensor::FRAME_W,
                       sensor::FRAME_H, model.vertices.data(), model.normals.data(), model.colors.data());
    model.pose = end_pose;
    sensor::buildFrameData(last_depth.data(), no_rgb.data(), fd, hp.min_depth, hp.max_depth, K);
    sensor::computeNormals(fd);
    sensor::FramePyramid pyr;
    sensor::buildFramePyramid(fd, pyr);
    tracking::ICPTracker icp;
    icp.setIntrinsics(K);
    const tracking::ICPResult loop = icp.track(pyr, model, end_pose, model.pose);
    // Only the observable part of the correction: facing bare walls, the solve's
    // answer along unobservable directions is arbitrary (same projection the
    // tracker applies every frame).
    const tracking::ObservableMotion loop_obs =
        tracking::keepObservableMotion(end_pose, loop.pose, loop.information);
    const Eigen::Matrix4f corr = end_pose.inverse() * loop_obs.pose;
    const double loop_t = corr.block<3,1>(0,3).norm();
    const double loop_r = Eigen::AngleAxisd(corr.block<3,3>(0,0).cast<double>()).angle() * 180.0 / M_PI;
    const bool loop_valid = loop.inliers > 5000 && loop.pose.allFinite();
    // Where the camera ended relative to where it started (world = first
    // camera). For a spin that returns to the starting view this is drift plus
    // however far the person was from the exact start; for a synthetic spin
    // it is the drift.
    const double end_t = end_pose.block<3,1>(0,3).norm();
    const double end_r = Eigen::AngleAxisd(end_pose.block<3,3>(0,0).cast<double>()).angle() * 180.0 / M_PI;

    // ---- loss episodes. The trace (closed by stop()) says which candidate
    // re-acquired, what integrated, and how long relocalizing frames took.
    Csv trace;
    readCsv(opt.out + "/trace.csv", trace);
    std::unordered_map<uint64_t, std::string> reloc_source;
    std::vector<uint64_t> integrated_ids;
    std::vector<double> reloc_ms;
    int keyframes = 0;
    for (size_t t = 0; t < trace.rows.size(); ++t) {
        const std::string& kind = trace.at(t, "kind");
        const uint64_t fid = std::strtoull(trace.at(t, "frame_id").c_str(), nullptr, 10);
        if (kind == "integ") {
            integrated_ids.push_back(fid);
        } else if (kind == "track") {
            if (!trace.at(t, "reloc_source").empty()) reloc_source[fid] = trace.at(t, "reloc_source");
            if (trace.at(t, "relocalizing") == "1") reloc_ms.push_back(std::strtod(trace.at(t, "ms_track").c_str(), nullptr));
            keyframes = std::max(keyframes, std::atoi(trace.at(t, "keyframes").c_str()));
        }
    }
    struct Episode {
        int lost_at = -1, reacquired_at = -1, lost_frames = 0;
        std::string source;
        double tilt = -1.0;
        int good_next30 = 0, next30 = 0;
        bool relost_within30 = false;
        int integrated_in_probation = 0;
        PoseErr err_at, err_after30;
    };
    constexpr int kAfter = 30;
    constexpr int kProbationFrames = 3;   // PipelineController::kReacquireProbation
    const int lost_state = static_cast<int>(app::PipelineState::TrackingLost);
    std::vector<Episode> episodes;
    for (size_t i = 1; i < records.size(); ++i) {
        if (records[i].state != lost_state || records[i - 1].state == lost_state) continue;
        Episode ep;
        ep.lost_at = static_cast<int>(i);
        size_t j = i;
        while (j < records.size() && records[j].state == lost_state) ++j;
        ep.lost_frames = static_cast<int>(j - i);
        if (j < records.size()) {
            ep.reacquired_at = static_cast<int>(j);
            const auto src = reloc_source.find(records[j].id);
            ep.source = src != reloc_source.end() ? src->second : "?";
            ep.tilt = tilt_of[j];
            for (size_t k = j; k < std::min(records.size(), j + kAfter); ++k) {
                ++ep.next30;
                ep.good_next30 += records[k].quality == 0;
                if (k > j && records[k].state == lost_state) ep.relost_within30 = true;
            }
            const uint64_t lo = records[i].id;
            const uint64_t hi = records[std::min(records.size() - 1, j + kProbationFrames - 1)].id;
            for (uint64_t fid : integrated_ids) ep.integrated_in_probation += (fid >= lo && fid <= hi);
            ep.err_at = ref_err[j];
            if (j + kAfter < records.size()) ep.err_after30 = ref_err[j + kAfter];
        }
        episodes.push_back(ep);
    }

    const double yaw_deg = yaw_total * 180.0 / M_PI;
    std::printf("\nframes %zu (%.1f s, %.1f fps): good %d, poor %d, failed %d, lost frames %d "
                "(first at frame %d), timeouts %d\n",
                records.size(), wall_s, records.size() / wall_s, good, poor, failed, lost, first_lost,
                timeouts);
    std::printf("yaw tracked about gravity: %.1f deg\n", yaw_deg);
    std::printf("tilt vs accelerometer: mean %.2f deg, max %.2f deg (axis signs %+.0f %+.0f %+.0f)\n",
                tilt_n ? tilt_sum / tilt_n : 0.0, tilt_max, best_sign.x(), best_sign.y(), best_sign.z());
    std::printf("loop closure (last frame vs first-%zu-frame model): %s%.1f mm / %.2f deg "
                "(inliers %d, %d unobservable dofs ignored)\n",
                first_depth.size(), loop_valid ? "" : "[UNRELIABLE] ", loop_t * 1e3, loop_r, loop.inliers,
                loop_obs.degenerate_dofs);
    std::printf("end pose vs start pose: %.1f mm / %.2f deg\n", end_t * 1e3, end_r);
    int reacquired = 0;
    for (const Episode& ep : episodes) reacquired += ep.reacquired_at >= 0;
    std::printf("loss episodes %zu, re-acquired %d; relocalizing frames %.1f ms p50 / %.1f ms p95; keyframes %d\n",
                episodes.size(), reacquired, percentile(reloc_ms, 0.5), percentile(reloc_ms, 0.95), keyframes);
    if (!episodes.empty()) {
        std::printf("  lost@  reacq@  frames  source               tilt  good/30  relost  integ  ref@reacq       ref@+30\n");
    }
    for (const Episode& ep : episodes) {
        std::printf("  %5d  %6d  %6d  %-19s %5.1f  %3d/%-3d  %6s  %5d  %6.1f mm %5.2f  %6.1f mm %5.2f\n", ep.lost_at,
                    ep.reacquired_at, ep.lost_frames, ep.reacquired_at >= 0 ? ep.source.c_str() : "(never)",
                    ep.tilt, ep.good_next30, ep.next30, ep.relost_within30 ? "yes" : "no",
                    ep.integrated_in_probation, ep.err_at.mm, ep.err_at.deg, ep.err_after30.mm,
                    ep.err_after30.deg);
    }
    std::printf("outputs: %s/{trace.csv,frames.csv,summary.json%s}\n", opt.out.c_str(),
                mesh_ok ? ",mesh.ply" : "");

    std::ofstream js(opt.out + "/summary.json");
    js << "{\n"
       << "  \"recording\": \"" << opt.dir << "\",\n"
       << "  \"backend\": \"" << backend_label << "\",\n"
       << "  \"preset\": \"" << opt.preset << "\",\n"
       << "  \"fx\": " << K.fx << ", \"intrinsics_source\": \"" << intr_source << "\",\n"
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
       << ", \"loop_unobservable_dofs\": " << loop_obs.degenerate_dofs << ",\n"
       << "  \"end_vs_start_mm\": " << end_t * 1e3 << ", \"end_vs_start_deg\": " << end_r << ",\n"
       << "  \"reloc_ms_p50\": " << percentile(reloc_ms, 0.5) << ", \"reloc_ms_p95\": " << percentile(reloc_ms, 0.95)
       << ", \"keyframes\": " << keyframes << ",\n"
       << "  \"episodes\": [";
    for (size_t k = 0; k < episodes.size(); ++k) {
        const Episode& ep = episodes[k];
        js << (k ? "," : "") << "\n    {\"lost_at\": " << ep.lost_at << ", \"reacquired_at\": " << ep.reacquired_at
           << ", \"lost_frames\": " << ep.lost_frames << ", \"source\": \"" << ep.source << "\", \"tilt_deg\": "
           << ep.tilt << ", \"good_next30\": " << ep.good_next30 << ", \"next30\": " << ep.next30
           << ", \"relost_within30\": " << (ep.relost_within30 ? "true" : "false")
           << ", \"integrated_in_probation\": " << ep.integrated_in_probation << ", \"ref_err_mm\": " << ep.err_at.mm
           << ", \"ref_err_deg\": " << ep.err_at.deg << ", \"ref_err_after30_mm\": " << ep.err_after30.mm
           << ", \"ref_err_after30_deg\": " << ep.err_after30.deg << "}";
    }
    js << (episodes.empty() ? "]" : "\n  ]") << "\n}\n";
    return 0;
}
