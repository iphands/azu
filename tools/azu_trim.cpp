// azu_trim: trim a fakenect recording to its still holds.
//
// Recording protocol (docs/RECORD_AND_REPLAY.md): start fakenect-record, get in
// position, hold still >= 10 s, do the capture, hold still >= 10 s, reach to stop.
// azu_trim finds the first and last hold (tools/Stillness.h: depth change AND
// accelerometer), prints a 0.5 s timeline of both ends for review, and writes a
// trimmed copy that keeps [start of the first hold, end of the last hold]: the
// positioning motion and the reach are cut, the holds are kept.
//
// The copy is a normal recording (INDEX.txt + the kept files as hardlinks, copies
// across filesystems, + device.json + trim.json); the original is not modified.
//
// usage: azu_trim <recording_dir> [--out DIR] [--dry-run] [--min-still S]
//                 [--search S] [--depth-thresh F] [--accel-thresh G]
//                 [--margin S] [--start S] [--end S]
#include "FakenectRecording.h"
#include "Stillness.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string dir, out;
    bool dry_run = false;
    double start = -1.0, end = -1.0;   // manual cut (s), -1 = detect
    azu_rec::StillOptions still;
};

void usage() {
    std::fprintf(stderr,
                 "usage: azu_trim <recording_dir> [--out DIR] [--dry-run] [--min-still S]\n"
                 "                [--search S] [--depth-thresh F] [--accel-thresh G]\n"
                 "                [--margin S] [--start S] [--end S]\n");
}

bool parseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto num = [&](const char* name) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return std::strtod(argv[++i], nullptr);
        };
        if (a == "--out" && i + 1 < argc) o.out = argv[++i];
        else if (a == "--dry-run") o.dry_run = true;
        else if (a == "--min-still") o.still.min_still = num("--min-still");
        else if (a == "--search") o.still.search = num("--search");
        else if (a == "--depth-thresh") o.still.depth_thresh = num("--depth-thresh");
        else if (a == "--accel-thresh") o.still.accel_thresh_g = num("--accel-thresh");
        else if (a == "--margin") o.still.margin = num("--margin");
        else if (a == "--start") o.start = num("--start");
        else if (a == "--end") o.end = num("--end");
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return false;
        } else o.dir = a;
    }
    while (o.dir.size() > 1 && o.dir.back() == '/') o.dir.pop_back();
    if (o.out.empty()) o.out = o.dir + "-trimmed";
    return !o.dir.empty();
}

// One row per 0.5 s: depth-change median, mean accelerometer std, share of
// still frames, and a marker where a cut falls.
void printTimeline(const std::vector<azu_rec::FrameMotion>& f, const std::vector<bool>& still,
                   double from, double to, double cut, const azu_rec::StillOptions& o) {
    std::printf("   time   depth   accel g  still  (depth # per %.3f, accel = per %.3f g)\n",
                o.depth_thresh / 10.0, o.accel_thresh_g / 10.0);
    for (double t = std::max(0.0, from); t < to; t += 0.5) {
        std::vector<double> d;
        double a = 0.0;
        int na = 0, ns = 0;
        for (size_t i = 0; i < f.size(); ++i) {
            if (f[i].t < t || f[i].t >= t + 0.5) continue;
            d.push_back(f[i].depth_change);
            if (f[i].has_accel) { a += f[i].accel_std_g; ++na; }
            ns += still[i] ? 1 : 0;
        }
        if (d.empty()) continue;
        std::sort(d.begin(), d.end());
        const double dm = d[d.size() / 2];
        const double am = na ? a / na : -1.0;
        const std::string dbar(static_cast<size_t>(std::min(30.0, dm / (o.depth_thresh / 10.0))), '#');
        const std::string abar(am < 0 ? 0 : static_cast<size_t>(std::min(30.0, am / (o.accel_thresh_g / 10.0))), '=');
        const bool here = cut >= t && cut < t + 0.5;
        std::printf("  %5.1fs  %.4f  %s  %3.0f%%  %-30s %-30s%s\n", t, dm,
                    am < 0 ? "  --  " : (std::to_string(am).substr(0, 6)).c_str(),
                    100.0 * ns / static_cast<double>(d.size()), dbar.c_str(), abar.c_str(),
                    here ? " << cut" : "");
    }
}

bool linkOrCopy(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    fs::create_hard_link(from, to, ec);
    if (!ec) return true;
    return fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec) && !ec;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        usage();
        return 2;
    }

    bool ok = false;
    const std::vector<azu_rec::Entry> entries = azu_rec::parseIndex(opt.dir, &ok);
    if (!ok) {
        std::fprintf(stderr, "no INDEX.txt in %s\n", opt.dir.c_str());
        return 1;
    }
    const auto first_depth = std::find_if(entries.begin(), entries.end(),
                                          [](const azu_rec::Entry& e) { return e.type == 'd'; });
    if (first_depth == entries.end()) {
        std::fprintf(stderr, "no depth frames in %s\n", opt.dir.c_str());
        return 1;
    }
    const double t0 = first_depth->host_s;

    // ---- signals
    std::vector<azu_rec::AccelSample> accel;
    for (const auto& e : entries) {
        if (e.type != 'a') continue;
        int16_t c[3];
        if (azu_rec::readAccelCounts(opt.dir, e, c)) {
            accel.push_back({e.host_s - t0, c[0] / azu_rec::kCountsPerG, c[1] / azu_rec::kCountsPerG,
                             c[2] / azu_rec::kCountsPerG});
        }
    }
    std::sort(accel.begin(), accel.end(),
              [](const azu_rec::AccelSample& a, const azu_rec::AccelSample& b) { return a.t < b.t; });

    std::vector<azu_rec::FrameMotion> frames;
    std::vector<uint16_t> prev, cur;
    size_t n_depth = 0;
    for (const auto& e : entries) n_depth += e.type == 'd';
    std::printf("azu_trim %s: %zu depth frames, %zu accelerometer samples\n", opt.dir.c_str(), n_depth,
                accel.size());
    for (const auto& e : entries) {
        if (e.type != 'd' || !azu_rec::readDepth(opt.dir, e, cur)) continue;
        azu_rec::FrameMotion m;
        m.t = e.host_s - t0;
        m.depth_change = prev.empty() ? 0.0
                                      : azu_rec::depthChange(prev, cur, azu_rec::kWidth, azu_rec::kHeight);
        const double s = azu_rec::accelStd(accel, m.t, 0.5 * opt.still.window);
        m.has_accel = s >= 0.0;
        m.accel_std_g = std::max(0.0, s);
        frames.push_back(m);
        prev.swap(cur);
    }
    if (frames.size() > 1) frames[0].depth_change = frames[1].depth_change;
    if (frames.size() < 2) {
        std::fprintf(stderr, "too few readable depth frames\n");
        return 1;
    }

    // ---- detect
    const azu_rec::StillOptions& so = opt.still;
    const azu_rec::TrimResult r = azu_rec::findTrim(frames, so);
    const std::vector<bool> still = azu_rec::stillFlags(frames, so);
    const double T = frames.back().t;
    const bool start_ok = r.start_found || opt.start >= 0.0;
    const bool end_ok = r.end_found || opt.end >= 0.0;
    const double start = opt.start >= 0.0 ? opt.start : r.start_t;
    const double end = opt.end >= 0.0 ? opt.end : r.end_t;

    std::printf("\n== first %.0f s\n", so.search);
    printTimeline(frames, still, 0.0, std::min(T, so.search), start_ok ? start : -1.0, so);
    std::printf("\n== last %.0f s\n", so.search);
    printTimeline(frames, still, std::max(0.0, T - so.search), T + 0.5, end_ok ? end : -1.0, so);

    auto frameAt = [&](double t) {
        size_t i = 0;
        while (i + 1 < frames.size() && frames[i].t < t) ++i;
        return i;
    };
    std::printf("\nrecording %.1f s\n", T);
    if (r.have_start_run) {
        std::printf("start hold %.2f-%.2f s (%.1f s)%s\n", r.start_run.t0, r.start_run.t1,
                    r.start_run.duration(), r.start_found ? "" : "  [shorter than --min-still]");
    } else {
        std::printf("start hold: none in the first %.0f s\n", so.search);
    }
    if (r.have_end_run) {
        std::printf("end hold   %.2f-%.2f s (%.1f s)%s\n", r.end_run.t0, r.end_run.t1, r.end_run.duration(),
                    r.end_found ? "" : "  [shorter than --min-still]");
    } else {
        std::printf("end hold: none in the last %.0f s\n", so.search);
    }
    if (!start_ok || !end_ok) {
        std::printf("\nno trim: %s%s%s not found with --min-still %.1f. Review the timeline, then\n"
                    "rerun with a shorter --min-still, other thresholds, or --start/--end seconds.\n",
                    start_ok ? "" : "start hold", (!start_ok && !end_ok) ? " and " : "",
                    end_ok ? "" : "end hold", so.min_still);
        return 3;
    }
    if (!(end > start)) {
        std::printf("\nno trim: end %.2f s is not after start %.2f s\n", end, start);
        return 3;
    }
    std::printf("cut %.2f-%.2f s (depth frames %zu-%zu of %zu), keeping %.1f s%s\n", start, end,
                frameAt(start), frameAt(end), frames.size(), end - start,
                (opt.start >= 0.0 || opt.end >= 0.0) ? "  [manual override]" : "");

    if (opt.dry_run) return 0;

    // ---- write the trimmed copy
    if (fs::exists(fs::path(opt.out) / "INDEX.txt")) {
        std::fprintf(stderr, "%s already holds a recording; not overwriting\n", opt.out.c_str());
        return 1;
    }
    std::error_code ec;
    fs::create_directories(opt.out, ec);
    std::ofstream index(fs::path(opt.out) / "INDEX.txt");
    size_t kept = 0, copied = 0;
    for (const auto& e : entries) {
        const double t = e.host_s - t0;
        if (t < start || t > end) continue;
        const fs::path from = fs::path(opt.dir) / e.file, to = fs::path(opt.out) / e.file;
        std::error_code lec;
        fs::create_hard_link(from, to, lec);
        if (lec) {
            if (!linkOrCopy(from, to)) {
                std::fprintf(stderr, "cannot link or copy %s\n", from.c_str());
                return 1;
            }
            ++copied;
        }
        index << e.file << '\n';
        ++kept;
    }
    index.close();
    if (fs::exists(fs::path(opt.dir) / "device.json")) {
        fs::copy_file(fs::path(opt.dir) / "device.json", fs::path(opt.out) / "device.json",
                      fs::copy_options::overwrite_existing, ec);
    }
    std::ofstream js(fs::path(opt.out) / "trim.json");
    js << "{\n  \"source\": \"" << opt.dir << "\",\n"
       << "  \"start_s\": " << start << ", \"end_s\": " << end << ",\n"
       << "  \"source_duration_s\": " << T << ",\n"
       << "  \"start_hold\": [" << r.start_run.t0 << ", " << r.start_run.t1 << "],\n"
       << "  \"end_hold\": [" << r.end_run.t0 << ", " << r.end_run.t1 << "],\n"
       << "  \"manual\": " << ((opt.start >= 0.0 || opt.end >= 0.0) ? "true" : "false") << ",\n"
       << "  \"min_still_s\": " << so.min_still << ", \"search_s\": " << so.search
       << ", \"depth_thresh\": " << so.depth_thresh << ", \"accel_thresh_g\": " << so.accel_thresh_g
       << ", \"margin_s\": " << so.margin << ", \"bridge_s\": " << so.bridge << "\n}\n";
    std::printf("wrote %s: %zu of %zu entries%s, device.json, trim.json\n", opt.out.c_str(), kept,
                entries.size(), copied ? " (some copied, not hardlinked)" : " (hardlinks)");
    return 0;
}
