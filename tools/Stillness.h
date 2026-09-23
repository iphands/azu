// Finding the still periods at the start and end of a handheld recording.
//
// Recording protocol (docs/RECORD_AND_REPLAY.md): start the recorder, get in
// position, hold still >= 10 s, do the capture, hold still >= 10 s, reach to stop.
// The trim keeps [start of the first hold, end of the last hold] and drops the
// positioning motion before it and the reach after it.
//
// A depth frame is STILL when both signals agree (calibrated on a real handheld
// take, 0.5 s windows):
//   depth: share of sampled pixels, valid in both frames, whose raw 11-bit code
//          changed by more than 3 since the previous frame; rolling 0.5 s median.
//          Handheld still 0.002-0.009, turning 0.03-0.25.
//   accel: std of the accelerometer vector over a centred 0.5 s window, in g.
//          Handheld still 0.013-0.022 g, turning 0.027-0.067 g, reaching for the
//          keyboard 0.07-0.09 g (a reach barely moves the camera, so depth alone
//          misses it).
// Holds are runs of still frames; gaps up to `bridge` s are bridged (a 0.2 s
// jolt marks ~0.7 s of frames through the 0.5 s accelerometer window) and a hold
// must last `min_still` s (a slow turn facing a flat wall can look still for
// ~1.5 s).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace azu_rec {

struct StillOptions {
    double depth_thresh   = 0.015;  // share of changed pixels
    double accel_thresh_g = 0.025;  // accelerometer std, g
    double min_still      = 3.0;    // s
    double search         = 20.0;   // s, from each end
    double margin         = 0.25;   // s, cut this far inside a hold
    double bridge         = 1.0;    // s, gaps inside a hold up to this are bridged
    double window         = 0.5;    // s, depth median / accelerometer std window
};

struct FrameMotion {
    double t = 0.0;              // s since the first depth frame
    double depth_change = 0.0;   // share of changed pixels vs the previous frame
    double accel_std_g = 0.0;
    bool   has_accel = false;
};

struct AccelSample {
    double t;           // s, same origin as FrameMotion::t
    double x, y, z;     // g
};

// Share of pixels (every `step`-th row and column) valid in both frames (raw
// 1..2046) whose code changed by more than `code_thresh`.
inline double depthChange(const std::vector<uint16_t>& prev, const std::vector<uint16_t>& cur,
                          int width, int height, int step = 4, int code_thresh = 3) {
    size_t both = 0, changed = 0;
    for (int y = 0; y < height; y += step) {
        for (int x = 0; x < width; x += step) {
            const size_t i = static_cast<size_t>(y) * width + x;
            const uint16_t a = prev[i], b = cur[i];
            if (a == 0 || a >= 2047 || b == 0 || b >= 2047) continue;
            ++both;
            if (std::abs(static_cast<int>(a) - static_cast<int>(b)) > code_thresh) ++changed;
        }
    }
    return both ? static_cast<double>(changed) / static_cast<double>(both) : 0.0;
}

// Std of the accelerometer vector over samples within +-half_window of t;
// samples sorted by t. Returns -1 when fewer than 3 samples fall in the window.
inline double accelStd(const std::vector<AccelSample>& s, double t, double half_window) {
    auto lo = std::lower_bound(s.begin(), s.end(), t - half_window,
                               [](const AccelSample& a, double v) { return a.t < v; });
    auto hi = std::upper_bound(s.begin(), s.end(), t + half_window,
                               [](double v, const AccelSample& a) { return v < a.t; });
    const long n = static_cast<long>(hi - lo);
    if (n < 3) return -1.0;
    double mx = 0, my = 0, mz = 0;
    for (auto it = lo; it != hi; ++it) { mx += it->x; my += it->y; mz += it->z; }
    mx /= n; my /= n; mz /= n;
    double v = 0;
    for (auto it = lo; it != hi; ++it) {
        v += (it->x - mx) * (it->x - mx) + (it->y - my) * (it->y - my) + (it->z - mz) * (it->z - mz);
    }
    return std::sqrt(v / n);
}

inline std::vector<bool> stillFlags(const std::vector<FrameMotion>& f, const StillOptions& o) {
    std::vector<bool> flags(f.size(), false);
    size_t lo = 0, hi = 0;
    std::vector<double> d;
    for (size_t i = 0; i < f.size(); ++i) {
        while (lo < f.size() && f[lo].t < f[i].t - 0.5 * o.window) ++lo;
        while (hi < f.size() && f[hi].t <= f[i].t + 0.5 * o.window) ++hi;
        d.clear();
        for (size_t k = lo; k < hi; ++k) d.push_back(f[k].depth_change);
        std::nth_element(d.begin(), d.begin() + static_cast<long>(d.size() / 2), d.end());
        const double depth_med = d[d.size() / 2];
        const bool accel_ok = !f[i].has_accel || f[i].accel_std_g < o.accel_thresh_g;
        flags[i] = depth_med < o.depth_thresh && accel_ok;
    }
    return flags;
}

struct Run {
    size_t i0 = 0, i1 = 0;   // first / last still frame index
    double t0 = 0.0, t1 = 0.0;
    double duration() const { return t1 - t0; }
};

// Runs of still frames, bridging non-still gaps of at most `bridge` seconds.
inline std::vector<Run> stillRuns(const std::vector<FrameMotion>& f, const std::vector<bool>& flags,
                                  const StillOptions& o) {
    std::vector<Run> runs;
    for (size_t i = 0; i < f.size(); ++i) {
        if (!flags[i]) continue;
        if (!runs.empty() && f[i].t - runs.back().t1 <= o.bridge + 1e-9) {
            runs.back().i1 = i;
            runs.back().t1 = f[i].t;
        } else {
            runs.push_back(Run{i, i, f[i].t, f[i].t});
        }
    }
    return runs;
}

struct TrimResult {
    bool   start_found = false, end_found = false;
    double start_t = 0.0, end_t = 0.0;   // cut times (s); valid when found
    Run    start_run, end_run;           // the hold used, or the best candidate
    bool   have_start_run = false, have_end_run = false;
};

// Start: the first hold >= min_still that begins within the first `search`
// seconds; end: the last hold >= min_still that ends within the last `search`
// seconds. When none qualifies, *_run is the longest candidate in that range.
inline TrimResult findTrim(const std::vector<FrameMotion>& f, const StillOptions& o) {
    TrimResult r;
    if (f.empty()) return r;
    const std::vector<Run> runs = stillRuns(f, stillFlags(f, o), o);
    const double t_begin = f.front().t, t_end = f.back().t;
    for (const Run& run : runs) {
        if (run.t0 > t_begin + o.search) break;
        if (run.duration() >= o.min_still) {
            r.start_found = true;
            r.start_run = run;
            r.have_start_run = true;
            break;
        }
        if (!r.have_start_run || run.duration() > r.start_run.duration()) {
            r.start_run = run;
            r.have_start_run = true;
        }
    }
    for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
        if (it->t1 < t_end - o.search) break;
        if (it->duration() >= o.min_still) {
            r.end_found = true;
            r.end_run = *it;
            r.have_end_run = true;
            break;
        }
        if (!r.have_end_run || it->duration() > r.end_run.duration()) {
            r.end_run = *it;
            r.have_end_run = true;
        }
    }
    if (r.start_found) r.start_t = std::min(r.start_run.t0 + o.margin, r.start_run.t1);
    if (r.end_found) r.end_t = std::max(r.end_run.t1 - o.margin, r.end_run.t0);
    if (r.start_found && r.end_found && r.end_t <= r.start_t) {
        r.start_found = r.end_found = false;   // one hold only: nothing to trim to
    }
    return r;
}

}  // namespace azu_rec
