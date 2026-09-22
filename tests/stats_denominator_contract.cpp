// stats_denominator_contract (big-fix todo 33): CPU-only contract for the three
// CSV diagnostic channels (src/tsdf/TSDFVolume.cpp AZU_TSDF_LOG,
// src/sensor/SignalConditioner_omp.cpp KFUSION_LOG, src/sensor/FrameData.cpp
// AZU_FRAME_LOG) and, through them, for the per-instance / documented-global
// statistics accumulators those channels publish. All three translation units are
// carried by azu_test_core; gate 15 in tests/CMakeLists.txt makes a local replica
// fail to link and scripts/test-cpu-big-fix.sh's nm -C guard catches a source-only
// replica.
//
// The bug class this locks is a mean whose denominator is the RASTER SIZE instead
// of the CONTRIBUTING SAMPLE COUNT, and its NaN twin (0/0 on an all-invalid
// frame). Every expectation is a property of a mean, or a property of the fixture,
// never a snapshot of product output:
//   * [TSDF] A mean of k non-negative samples with maximum M satisfies M/k <= mean
//     <= mean*k, i.e. mean * count >= maximum and mean <= maximum. The fixture is a
//     frame with exactly ONE valid pixel, so the contributor count is the raster
//     ray traversal (<= resolution voxels) while the raster is 307200 pixels; a
//     raster-size denominator divides the mean by ~44000 and fails mean*count >= max
//     by four orders of magnitude. depth_filtered is an independent count of the
//     fixture's own invalid pixels, and an all-invalid frame must report avg 0.0
//     exactly, never NaN.
//   * [TSDF] Padding invariance: identical valid pixel sets must yield identical
//     counters and identical means whether the remaining pixels carry the invalid
//     sentinel or an out-of-band distance.
//   * [SignalConditioner] The EMA fixture is UNIFORM over a sub-rectangle, so every
//     contributing pixel contributes the SAME delta and the mean must equal the
//     maximum EXACTLY; ema_filtered must equal the fixture's in-band pixel count.
//     A raster denominator yields max*K/307200 = max/61 and fails. A frame with no
//     contributing pixel must report 0.0, never NaN, and a repeat of the same frame
//     must report a zero mean.
//   * [FrameData] A UNIFORM valid rectangle makes avg_depth equal max_depth exactly
//     and both equal the fixture's single depth value; vertices_computed and
//     depth_filtered must partition the raster exactly.
//   * [all] The channels are env-gated and fail-closed: with the variable unset the
//     product path runs and writes NO file at all.
// Each scenario needs a fresh env and a fresh working directory, and the sinks are
// write-once by construction, so every scenario runs in its own child process
// (fork + execve of /proc/self/exe) inside a private temp directory that the
// parent inspects and removes. No device, display, GPU, OpenGL context, sensor,
// network or wall-clock oracle.

#include "sensor/DepthValidity.h"
#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "sensor/SignalConditioner.h"
#include "tsdf/TSDFVolume.h"
#include "utils/Logger.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "stats_denominator_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::sensor::FRAME_H;
using kfusion::sensor::FRAME_W;
using kfusion::sensor::FrameData;
using kfusion::sensor::SignalConditioner;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;

constexpr int kRasters = FRAME_W * FRAME_H;

// The fixture rectangle: a strict sub-rectangle of the raster, so K < raster size
// and a raster-size denominator is arithmetically distinguishable from a
// contributing-count denominator.
constexpr int kX0 = 100, kX1 = 200, kY0 = 100, kY1 = 150;
constexpr int kK  = (kX1 - kX0) * (kY1 - kY0);

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),        \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

std::string g_dir;  // scenario working directory (child and parent agree on it)


// True when the current working directory holds any CSV at all.
struct DirCsvCounter {
    int count = 0;
    explicit DirCsvCounter(const std::string& dir) {
        DIR* d = ::opendir(dir.c_str());
        if (!d) return;
        while (dirent* e = ::readdir(d)) {
            const std::string n(e->d_name);
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".csv") == 0) ++count;
        }
        ::closedir(d);
    }
};

bool anyCsvInDir() {
    char cwd[4096];
    if (!::getcwd(cwd, sizeof(cwd))) return false;
    return DirCsvCounter(cwd).count != 0;
}

bool inRect(int x, int y) { return x >= kX0 && x < kX1 && y >= kY0 && y < kY1; }

// ------------------------------------------------------------- CSV reader
struct Csv {
    std::vector<std::string> rows;   // data rows only, header removed
    std::string header;
    bool ok = false;
};

Csv readCsv(const std::string& name) {
    Csv out;
    std::ifstream in(name);
    if (!in.is_open()) return out;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (first) { out.header = line; first = false; continue; }
        out.rows.push_back(line);
    }
    out.ok = true;
    return out;
}

// Column access is by NAME from the header, so a reordered column is a loud
// failure here rather than a silently mis-read field.
int columnIndex(const Csv& csv, const std::string& name) {
    std::string h = csv.header;
    if (h.size() > 4 && h.compare(0, 4, "PK\x03\x04") == 0) return -1;
    size_t pos = 0;
    int i = 0;
    while (pos <= h.size()) {
        const auto comma = h.find(',', pos);
        const std::string field =
            h.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (field == name) return i;
        if (comma == std::string::npos) break;
        pos = comma + 1;
        ++i;
    }
    return -1;
}

double field(const Csv& csv, size_t row, const std::string& name) {
    const int col = columnIndex(csv, name);
    if (col < 0 || row >= csv.rows.size()) return std::nan("");
    std::string cell = csv.rows[row];
    size_t pos = 0;
    for (int i = 0; i < col; ++i) {
        const auto comma = cell.find(',', pos);
        if (comma == std::string::npos) return std::nan("");
        pos = comma + 1;
    }
    const auto comma = cell.find(',', pos);
    const std::string value =
        cell.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    try {
        return std::stod(value);
    } catch (...) {
        return std::nan("");
    }
}

// ------------------------------------------------------------- scenarios

// A mean of non-negative samples obeys max/count <= mean <= max. Both halves are
// asserted together, because the pair is exactly "divided by the contributors".
void checkMeanBounds(double avg, double max_v, double count, const char* what) {
    if (count <= 0.0) {
        CHECK(avg == 0.0, what);
        return;
    }
    CHECK(!std::isnan(avg) && !std::isinf(avg), what);
    CHECK(avg <= max_v + 1e-6, "mean never exceeds the maximum");
    CHECK(avg * count >= max_v - 1e-6, "mean * contributing count covers the maximum");
}

void scenarioTsdf() {
    static constexpr float kMinD = 0.3f;
    static constexpr float kMaxD = 5.0f;

    TSDFParams p;
    p.resolution = 64;
    p.voxel_size = 0.02f;
    p.origin     = Eigen::Vector3f(-0.64f, -0.64f, 0.0f);

    std::vector<float> depth(static_cast<size_t>(kRasters), 0.0f);
    const int single = 240 * FRAME_W + 320;
    depth[single] = 1.0f;

    // One integrate() from a fresh volume: the CSV row is that frame's whole story.
    {
        TSDFVolume v(p);
        v.integrate(depth.data(), nullptr, Eigen::Matrix4f::Identity(),
                    kfusion::sensor::FX, kfusion::sensor::FY,
                    kfusion::sensor::CX, kfusion::sensor::CY,
                    FRAME_W, FRAME_H, kMinD, kMaxD);
    }
    const Csv csv = readCsv("tsdf_integration_log.csv");
    CHECK(csv.ok, "AZU_TSDF_LOG=1 creates tsdf_integration_log.csv");
    CHECK(csv.rows.size() == 1, "one integrate() emits exactly one CSV row");
    if (!csv.ok || csv.rows.empty()) return;

    const double filtered = field(csv, 0, "depth_filtered");
    const double updated  = field(csv, 0, "voxels_updated");
    const double max_sdf  = field(csv, 0, "max_sdf");
    const double avg_sdf  = field(csv, 0, "avg_sdf");

    CHECK(filtered == kRasters - 1, "depth_filtered counts the fixture's invalid pixels");
    CHECK(updated >= 1.0, "a valid pixel updates at least one voxel");
    CHECK(updated <= p.resolution,
          "one pixel cannot update more voxels than one ray traversal");
    // The decisive pair. With a raster-size denominator avg_sdf is smaller than
    // this by ~44000/updated, so the inequality fails loudly.
    checkMeanBounds(avg_sdf, max_sdf, updated, "TSDF avg_sdf is a mean over updated voxels");
    CHECK(avg_sdf > 1e-4, "avg_sdf is not diluted by the 307200-pixel raster");

    // Padding invariance: same valid set, the rest out-of-band instead of the
    // invalid sentinel. Every counter and the mean must be untouched.
    std::vector<float> padded(static_cast<size_t>(kRasters), 10.0f);
    padded[single] = 1.0f;
    {
        TSDFVolume v(p);
        v.integrate(padded.data(), nullptr, Eigen::Matrix4f::Identity(),
                    kfusion::sensor::FX, kfusion::sensor::FY,
                    kfusion::sensor::CX, kfusion::sensor::CY,
                    FRAME_W, FRAME_H, kMinD, kMaxD);
    }
    const Csv pad = readCsv("tsdf_integration_log.csv");
    CHECK(pad.rows.size() == 2, "the padded frame appends a second row");
    if (pad.rows.size() == 2) {
        CHECK(field(pad, 1, "depth_filtered") == filtered, "padding does not change filtering");
        CHECK(field(pad, 1, "voxels_updated") == updated, "padding does not change contributors");
        CHECK(field(pad, 1, "max_sdf") == max_sdf, "padding does not change the maximum");
        CHECK(field(pad, 1, "avg_sdf") == avg_sdf, "padding does not change the mean");
    }

    // Zero contributors: the mean is undefined and must publish 0.0, never NaN.
    {
        TSDFVolume v(p);
        std::vector<float> none(static_cast<size_t>(kRasters), 0.0f);
        v.integrate(none.data(), nullptr, Eigen::Matrix4f::Identity(),
                    kfusion::sensor::FX, kfusion::sensor::FY,
                    kfusion::sensor::CX, kfusion::sensor::CY,
                    FRAME_W, FRAME_H, kMinD, kMaxD);
    }
    const Csv empty = readCsv("tsdf_integration_log.csv");
    CHECK(empty.rows.size() == 3, "the empty frame appends a third row");
    if (empty.rows.size() == 3) {
        CHECK(field(empty, 2, "voxels_updated") == 0, "no contributor is counted");
        CHECK(field(empty, 2, "avg_sdf") == 0.0, "a zero-contribution mean is exactly 0.0");
    }
}

void scenarioSignal() {
    // The CSV row is produced by the product's per-frame stage list, so the
    // fixture drives processCpu() rather than the EMA seam alone: an EMA-only call
    // updates the accumulator but no row is published, and a row is what this
    // contract reads back.
    static constexpr uint16_t kRawA = 500;      // in band
    static constexpr uint16_t kRawB = 460;      // in band, closer
    static constexpr uint16_t kSentinel = 2047; // invalid AND non-fillable
    static constexpr float kMinD = 0.3f;
    static constexpr float kMaxD = 2.5f;

    // A FULLY uniform depth raster keeps every earlier stage of the pipeline
    // (spatial median, hole fill, guided filter) output-equal to its input, so all
    // contributing pixels of the temporal stage carry the SAME delta. That is what
    // makes "mean == maximum" an exact, product-independent expectation.
    // In-band rectangle, strictly smaller than the raster, surrounded by the raw
    // 2047 sentinel (invalid but NOT fillable), so the contributor count is well
    // below the raster size while every earlier stage of the pipeline still leaves
    // the in-band field untouched.
    static constexpr int kSx0 = 200, kSx1 = 440, kSy0 = 160, kSy1 = 320;
    static constexpr int kSx  = kSx1 - kSx0, kSy = kSy1 - kSy0;
    static constexpr int kInBand = kSx * kSy;
    CHECK(kInBand * 4 <= kRasters, "the EMA fixture is well under one raster of contributors");

    auto frame_of = [&](uint16_t code, uint64_t id) {
        kfusion::sensor::RawFrame raw;
        CHECK(static_cast<int>(raw.depth.size()) == kRasters,
              "RawFrame depth raster matches the fixture raster");
        std::fill(raw.depth.begin(), raw.depth.end(), kSentinel);
        for (int y = kSy0; y < kSy1; ++y) {
            for (int x = kSx0; x < kSx1; ++x) {
                raw.depth[static_cast<size_t>(y * FRAME_W + x)] = code;
            }
        }
        std::fill(raw.rgb.begin(), raw.rgb.end(), uint8_t{0});
        raw.frame_id = id;
        return raw;
    };

    auto a = frame_of(kRawA, 1);
    auto b = frame_of(kRawB, 2);
    auto z = frame_of(kSentinel, 3);

    SignalConditioner sc;
    sc.process(a, nullptr, kMinD, kMaxD);              // every pixel is a fresh reset
    sc.process(a, nullptr, kMinD, kMaxD);              // identical frame: zero delta
    sc.process(b, nullptr, kMinD, kMaxD);              // uniform transition
    sc.process(z, nullptr, kMinD, kMaxD);              // no contributor at all

    const Csv csv = readCsv("signal_conditioner_log.csv");
    CHECK(csv.ok, "KFUSION_LOG=1 creates signal_conditioner_log.csv");
    CHECK(csv.rows.size() == 4, "four process() calls emit four rows");
    if (!csv.ok || csv.rows.size() < 4) return;

    // Stage accounting, from the fixture's own shape: both RGB smoothing stages and
    // the spatial depth median run on the whole raster, nothing is fillable, no
    // depth gradient exists, and every in-band pixel passes the guided filter.
    for (size_t r = 0; r < 3; ++r) {
        CHECK(field(csv, r, "bilateral_filtered") == kRasters,
              "the RGB bilateral stage counts the raster once");
        CHECK(field(csv, r, "median_filtered") == 2 * kRasters,
              "the RGB and depth median stages count the raster once each");
        CHECK(field(csv, r, "hole_filled") == 0, "a uniform raster has no fillable hole");
        // The gradient helper flags the outermost image ring as an edge pixel by
        // definition (it has no full stencil), and a uniform interior has zero
        // gradient, so the ring count is the fixture-derived expectation.
        // The gradient helper flags any pixel whose 4-neighbour stencil is not
        // fully usable, so exactly the inner rect's one-pixel perimeter band is an
        // edge and the untouched interior is not.
        CHECK(field(csv, r, "edge_pixels") == 2 * kSx + 2 * kSy - 4,
              "a uniform field flags exactly its stencil perimeter");
        CHECK(field(csv, r, "guided_filtered") == kInBand,
              "every in-band pixel passes the guided filter");
    }

    for (size_t r = 0; r < 3; ++r) {
        const double filtered = field(csv, r, "ema_filtered");
        const double max_d    = field(csv, r, "max_depth_delta");
        const double avg_d    = field(csv, r, "avg_depth_delta");
        CHECK(filtered == kInBand, "ema_filtered counts every in-band pixel");
        // UNIFORM fixture: every contributing delta is the same number, so the mean
        // equals the maximum to the last printed digit. A raster denominator leaves
        // max*kInBand/raster (an eighth of it here) and cannot satisfy this.
        CHECK(avg_d == max_d, "uniform deltas make the EMA mean equal its maximum");
        checkMeanBounds(avg_d, max_d, filtered, "EMA mean is bounded by its contributors");
    }

    // Every invalid pixel resets too, so a first frame resets the whole raster and
    // a repeated frame resets exactly its invalid surround.
    CHECK(field(csv, 0, "ema_reset") == kRasters, "a first frame resets every pixel");
    CHECK(field(csv, 1, "ema_reset") == kRasters - kInBand,
          "a repeated frame resets only its invalid surround");
    CHECK(field(csv, 1, "max_depth_delta") == 0, "an identical frame has zero delta");
    CHECK(field(csv, 1, "avg_depth_delta") == 0, "a zero-delta frame has zero mean");
    CHECK(field(csv, 2, "max_depth_delta") > 0.0, "a changed frame has a nonzero delta");

    CHECK(field(csv, 3, "ema_filtered") == 0, "an all-invalid frame has no contributor");
    CHECK(field(csv, 3, "ema_reset") == kRasters, "an all-invalid frame resets the raster");
    CHECK(field(csv, 3, "avg_depth_delta") == 0.0,
          "a zero-contribution EMA mean is exactly 0.0, not NaN");

    // Per-instance state, first through the temporal view: a SECOND conditioner
    // driven with the same raster must treat every pixel as brand new, i.e. its
    // state equals the frame's own depth rather than the history the first object
    // built up. This is the owning-object half of the todo's stats refactor.
    {
        SignalConditioner other;
        auto fresh = frame_of(kRawA, 20);
        other.applyDepthEmaForTests(fresh.depth, kMinD, kMaxD);
        const double expected =
            static_cast<double>(kfusion::sensor::cpuDepthMeters(kRawA, kMinD, kMaxD));
        const std::vector<float>& state = other.emaStateForTests();
        bool all_fresh = state.size() == fresh.depth.size();
        for (int y = kSy0; y < kSy1 && all_fresh; ++y) {
            for (int x = kSx0; x < kSx1; ++x) {
                const float v = state[static_cast<size_t>(y * FRAME_W + x)];
                if (std::fabs(static_cast<double>(v) - expected) > 1e-6) { all_fresh = false; break; }
            }
        }
        CHECK(all_fresh, "a second conditioner starts with no temporal history");
    }

    // Per-instance accounting, through the published rows: two conditioners
    // alternately emit one full frame and one fully-invalid frame. Each row must
    // carry ONLY its own frame's counters. A process-wide accumulator cannot keep
    // the invalid frame's row at zero once the other object has counted a raster of
    // contributors, so the alternating pair is the discriminator.
    {
        SignalConditioner other;
        auto full = frame_of(kRawA, 10);
        auto void_frame = frame_of(kSentinel, 11);
        other.reset();
        sc.process(full, nullptr, kMinD, kMaxD);
        other.process(void_frame, nullptr, kMinD, kMaxD);
        sc.process(full, nullptr, kMinD, kMaxD);
        other.process(void_frame, nullptr, kMinD, kMaxD);

        const Csv pair = readCsv("signal_conditioner_log.csv");
        CHECK(pair.rows.size() == 8, "the alternating frames append four more rows");
        if (pair.rows.size() == 8) {
            // Rows 4 and 6 belong to the full-frame object, rows 5 and 7 to the
            // invalid-frame object. Each must describe only its own frame: a
            // process-wide accumulator cannot leave rows 5/7 at zero after rows
            // 4/6 counted a rectangle full of contributors.
            CHECK(field(pair, 4, "ema_filtered") == kInBand,
                  "full-frame row one carries its own contributors");
            CHECK(field(pair, 6, "ema_filtered") == kInBand,
                  "full-frame row two carries its own contributors");
            CHECK(field(pair, 5, "ema_filtered") == 0,
                  "invalid-frame row one is not contaminated by the other object");
            CHECK(field(pair, 7, "ema_filtered") == 0,
                  "invalid-frame row two is not contaminated by the other object");
            CHECK(field(pair, 5, "guided_filtered") == 0 &&
                  field(pair, 7, "guided_filtered") == 0,
                  "invalid frames count no guided pixels");
            CHECK(field(pair, 5, "median_filtered") == 2 * kRasters,
                  "an invalid frame still runs both median stages");
        }
    }
}

void scenarioFrame() {
    static constexpr uint16_t kRaw = 500;
    static constexpr float kMinD = 0.3f;
    static constexpr float kMaxD = 2.5f;

    std::vector<uint16_t> raw(static_cast<size_t>(kRasters), 0);
    std::vector<uint8_t> rgb(static_cast<size_t>(kRasters) * 3, 0);
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            if (inRect(x, y)) raw[static_cast<size_t>(y * FRAME_W + x)] = kRaw;
        }
    }

    FrameData frame;
    kfusion::sensor::buildFrameData(raw.data(), rgb.data(), frame, kMinD, kMaxD);
    const Csv csv = readCsv("framedata_log.csv");
    CHECK(csv.ok, "AZU_FRAME_LOG=1 creates framedata_log.csv");
    CHECK(csv.rows.size() == 1, "one buildFrameData emits exactly one row");
    if (!csv.ok || csv.rows.empty()) return;

    const double filtered = field(csv, 0, "depth_filtered");
    const double verts    = field(csv, 0, "vertices_computed");
    const double max_d    = field(csv, 0, "max_depth");
    const double avg_d    = field(csv, 0, "avg_depth");

    CHECK(filtered + verts == kRasters, "the two counters partition the raster");
    CHECK(verts == kK, "vertices_computed counts the fixture's in-band rectangle");
    CHECK(filtered == kRasters - kK, "depth_filtered counts the complement");
    // The rectangle is UNIFORM in depth, so the mean over contributors is that one
    // depth value and therefore equal to the maximum exactly. Diluted by the raster
    // the mean is max*K/307200 and the equality fails.
    CHECK(avg_d == max_d, "uniform depth makes avg_depth equal max_depth");
    checkMeanBounds(avg_d, max_d, verts, "avg_depth is a mean over contributing pixels");
    CHECK(avg_d > 1e-3, "avg_depth is not diluted by the 307200-pixel raster");
    // Second, independent observation channel: the frame buffer the product just
    // published. The test computes the mean over the pixels the buffer itself
    // reports valid; the CSV must agree (tolerance covers only the 6-significant-
    // digit float the CSV prints and the accumulation order the OpenMP critical
    // sections leave free).
    double buf_sum = 0.0;
    int    buf_cnt = 0;
    for (const float d : frame.depth_meters) {
        if (d > 0.0f) { buf_sum += static_cast<double>(d); ++buf_cnt; }
    }
    CHECK(buf_cnt == kK, "the published buffer marks exactly the fixture rectangle valid");
    CHECK(std::fabs(avg_d - buf_sum / buf_cnt) <= 1e-5 * buf_sum / buf_cnt,
          "avg_depth matches the mean over the published buffer");
    CHECK(field(csv, 0, "normals_computed") == 0, "buildFrameData counts no normals");

    // An all-invalid raster: zero contributors, and the mean stays a real 0.0.
    {
        std::vector<uint16_t> none(static_cast<size_t>(kRasters), 0);
        FrameData empty_frame;
        kfusion::sensor::buildFrameData(none.data(), rgb.data(), empty_frame, kMinD, kMaxD);
    }
    const Csv empty = readCsv("framedata_log.csv");
    CHECK(empty.rows.size() == 2, "the empty frame appends a second row");
    if (empty.rows.size() == 2) {
        CHECK(field(empty, 1, "vertices_computed") == 0, "no vertex is counted");
        CHECK(field(empty, 1, "depth_filtered") == kRasters, "the whole raster is filtered");
        CHECK(field(empty, 1, "avg_depth") == 0.0, "a zero-contribution mean is exactly 0.0");
    }
}


// Every channel is env-gated. With the variables absent from the child's
// environment the same three product paths must run and create nothing at all: a
// diagnostic facility that opens its file unconditionally is a stray-file bug in
// the user's working directory, which is what this scenario rules out.
void scenarioOff() {
    TSDFParams p;
    p.resolution = 32;
    p.voxel_size = 0.04f;
    p.origin     = Eigen::Vector3f(-0.64f, -0.64f, 0.0f);
    std::vector<float> depth(static_cast<size_t>(kRasters), 1.0f);
    {
        TSDFVolume v(p);
        v.integrate(depth.data(), nullptr, Eigen::Matrix4f::Identity(),
                    kfusion::sensor::FX, kfusion::sensor::FY,
                    kfusion::sensor::CX, kfusion::sensor::CY,
                    FRAME_W, FRAME_H, 0.3f, 2.5f);
    }
    std::vector<uint16_t> raw(static_cast<size_t>(kRasters), 500);
    {
        SignalConditioner sc;
        sc.applyDepthEmaForTests(raw, 0.3f, 2.5f);
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(kRasters) * 3, 0);
    {
        FrameData frame;
        kfusion::sensor::buildFrameData(raw.data(), rgb.data(), frame, 0.3f, 2.5f);
    }
    CHECK(!anyCsvInDir(), "no diagnostic env var means no CSV file is created");
}


// ------------------------------------------------------------------ harness
// The sinks read their env and open their relative file exactly once per process,
// so a single binary cannot test both the enabled and the disabled state. Each
// scenario therefore runs in a fresh child process with an explicitly built
// environment inside a private directory the parent inspects.
const char* const kScenarios[] = {"off", "tsdf", "sig", "frame"};

std::string exePath() {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    return std::string(buf);
}

int runScenario(const std::string& name) {
    if (name == "off")   scenarioOff();
    if (name == "tsdf")  scenarioTsdf();
    if (name == "sig")   scenarioSignal();
    if (name == "frame") scenarioFrame();
    std::printf("stats_denominator_contract[%s]: %d checks, %d failures\n",
                name.c_str(), g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

bool firstLineHasPrefix(const std::string& path, const std::string& prefix) {
    std::ifstream in(path);
    std::string line;
    return std::getline(in, line) && line.rfind(prefix, 0) == 0;
}

int parentMain() {
    int failures = 0;
    char tmpl[64];
    std::snprintf(tmpl, sizeof(tmpl), "/tmp/azu_stats_XXXXXX");
    const std::string root = ::mkdtemp(tmpl) ? std::string(tmpl) : std::string();
    CHECK(!root.empty(), "a private temp root was created");
    if (root.empty()) return 1;

    const std::string exe = exePath();
    CHECK(!exe.empty(), "/proc/self/exe resolves");

    struct Case { const char* name; const char* var; const char* file; const char* header; };
    const Case cases[] = {
        {"off",   nullptr,                nullptr, nullptr},
        {"tsdf",  "AZU_TSDF_LOG=1",       "tsdf_integration_log.csv",
         "frame,depth_filtered,voxels_updated,color_updates,truncation_clamped,max_sdf,avg_sdf"},
        {"sig",   "KFUSION_LOG=1",        "signal_conditioner_log.csv",
         "frame,bilateral_filtered,median_filtered,hole_filled,guided_filtered,"},
        {"frame", "AZU_FRAME_LOG=1",      "framedata_log.csv",
         "frame,depth_filtered,vertices_computed,normals_computed,"},
    };

    for (const Case& c : cases) {
        std::string dir = root + "/" + c.name;
        CHECK(::mkdir(dir.c_str(), 0700) == 0, "scenario directory created");

        const pid_t pid = ::fork();
        if (pid == 0) {
            if (::chdir(dir.c_str()) != 0) _exit(120);
            std::vector<std::string> env_storage;
            env_storage.push_back(std::string("AZU_SCENARIO=") + c.name);
            env_storage.push_back(std::string("AZU_SCENARIO_DIR=") + dir);
            if (c.var) env_storage.push_back(c.var);
            std::vector<char*> envp;
            for (auto& e : env_storage) envp.push_back(const_cast<char*>(e.c_str()));
            envp.push_back(nullptr);
            std::vector<std::string> argv_storage = {exe, "--scenario", c.name, dir};
            std::vector<char*> argv;
            for (auto& a : argv_storage) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            ::execve(exe.c_str(), argv.data(), envp.data());
            _exit(127);
        }
        CHECK(pid > 0, "scenario child was spawned");
        int status = 0;
        CHECK(::waitpid(pid, &status, 0) == pid, "scenario child was reaped");
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "scenario child reported zero failures");

        if (c.file == nullptr) {
            DirCsvCounter counter(dir);
            CHECK(counter.count == 0, "the disabled scenario left no CSV behind");
        } else {
            const std::string path = dir + "/" + c.file;
            struct stat st{};
            CHECK(::stat(path.c_str(), &st) == 0 && st.st_size > 0,
                  "the enabled scenario wrote its CSV");
            CHECK(firstLineHasPrefix(path, c.header), "the CSV header column set is intact");
        }
        // Leave nothing behind: the scenario directories belong to this process.
        if (c.file) ::unlink((dir + "/" + c.file).c_str());
        ::rmdir(dir.c_str());
    }

    ::rmdir(root.c_str());
    std::printf("stats_denominator_contract: %d parent checks, %d failures\n",
                g_checks, failures + g_failures);
    const int total = failures + g_failures;
    std::printf("stats_denominator_contract: %s\n", total == 0 ? "PASS" : "FAIL");
    return total == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--scenario") == 0) {
        if (argc >= 4 && ::chdir(argv[3]) != 0) {
            std::printf("FAIL: scenario child could not enter its directory\n");
            return 121;
        }
        return runScenario(argv[2]);
    }
    return parentMain();
}
