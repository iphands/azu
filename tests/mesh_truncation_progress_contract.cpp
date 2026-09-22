// mesh_truncation_progress_contract (big-fix Todo 17): CPU-only contract for the
// triangle budget and the progress-callback thread-safety of
// MarchingCubes::extract. Public CPU API only (TSDFVolume + MarchingCubes::extract
// + setMaxTriangles); no device, display, GPU, sensor, filesystem or network.
//
// Canonical rules under test (docs/CANONICAL_SEMANTICS.md, "Marching Cubes" +
// include/meshing/MarchingCubes.h ProgressCallback contract):
//   P1 when the triangle budget is reached the extractor stops at a full triangle
//      boundary, sets MeshData::truncated, emits no partial triangle, and the mesh
//      still passes validate() with color lockstep intact;
//   P2 the callback is invoked only from the calling thread, never concurrently,
//      with finite values in [0, 1], nondecreasing, final call exactly 1.0;
//   P3 the emitted progress sequence is a pure function of the volume: two runs and
//      any OpenMP thread count produce the identical sequence;
//   P4 a degenerate (tiny) resolution is handled gracefully: no crash and no
//      non-finite progress, instead of dividing by a (resolution - 2) span.
//
// P1/P2/P3/P4 are the Todo 17 fail-before-fix witnesses: the pre-fix CPU path had no
// budget at all, called the callback from inside the OpenMP region (racy, order- and
// thread-dependent, and NaN progress at resolution 2).

#include "meshing/MarchingCubes.h"
#include "tsdf/TSDFVolume.h"
#include "utils/ColorMath.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

namespace {

using kfusion::meshing::MarchingCubes;
using kfusion::meshing::MeshData;
using kfusion::meshing::ProgressCallback;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;
using kfusion::utils::srgbUint8ToFloat;   // uint8 sRGB -> the volume's float sRGB domain

int g_failures = 0;
int g_sections = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s  [%s:%d]\n", (what), __FILE__, __LINE__);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

TSDFParams makeParams(int resolution, float voxel_size, float trunc) {
    TSDFParams p;
    p.resolution = resolution; p.voxel_size = voxel_size; p.truncation = trunc;
    p.max_weight = 8.0f; p.origin = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    return p;
}

// A fully observed analytic sphere that yields well over a hundred triangles, so a
// small budget genuinely bites.
void fillSphere(TSDFVolume& vol, int R, float vs, float rad) {
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const float t = ((vol.voxelToWorld(x, y, z) - center).norm() - rad) / p.truncation;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = std::max(-1.0f, std::min(1.0f, t));
                v.weight = p.max_weight;
                v.r = srgbUint8ToFloat(60); v.g = srgbUint8ToFloat(120); v.b = srgbUint8ToFloat(180);
            }
}

// ---------------------------------------------------------------------------
// Section 1 (RED): the triangle budget. A small cap must stop at a full-triangle
// boundary, flag truncation, keep colors in lockstep, and still validate.
// ---------------------------------------------------------------------------
void section_truncation() {
    ++g_sections;
    const int   R  = 24;
    const float vs = 0.04f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    fillSphere(vol, R, vs, 0.26f);

    MarchingCubes mc;
    const size_t cap = 50;
    mc.setMaxTriangles(cap);
    auto m = mc.extract(vol);
    CHECK(m != nullptr && !m->empty(), "s1: bounded extraction produced a mesh");
    if (!m || m->empty()) return;

    std::printf("s1: cap=%zu tris=%zu verts=%zu truncated=%d colors=%zu\n",
                cap, m->triangleCount(), m->positions.size(), m->truncated ? 1 : 0, m->colors.size());
    CHECK(m->triangleCount() <= cap,
          "s1: emitted triangles never exceed the budget");
    CHECK(m->truncated, "s1: a budget-limited extraction sets MeshData::truncated");
    CHECK(m->indices.size() % 3 == 0, "s1: no partial triangle row is emitted");
    CHECK(m->triangleCount() >= 1, "s1: at least one full triangle survives");
    CHECK(m->colors.size() == m->positions.size() * 3,
          "s1: colors stay in exact lockstep on the truncated mesh");
    CHECK(m->validate(nullptr), "s1: truncated mesh passes MeshData::validate()");
}

// Recorder that is safe even if the callback is (incorrectly) invoked concurrently,
// and that measures the maximum number of callbacks in flight at once. State is
// shared through a shared_ptr so the ProgressCallback target stays copy-constructible
// (std::function requires that), while the mutex/atomic live in the heap object.
struct ProgressState {
    std::mutex         mtx;
    std::vector<float> seq;
    std::atomic<int>   inflight{0};
    std::atomic<int>   max_inflight{0};
};

ProgressCallback makeProbe(const std::shared_ptr<ProgressState>& st) {
    return [st](float v) {
        const int now = st->inflight.fetch_add(1) + 1;
        int prev = st->max_inflight.load();
        while (now > prev && !st->max_inflight.compare_exchange_weak(prev, now)) {}
        // Widen the overlap window so concurrent invocations (the CPU defect) are
        // observed rather than accidentally serialized by timing.
        volatile int spin = 0;
        for (int i = 0; i < 4000; ++i) spin += i;
        (void)spin;
        st->inflight.fetch_sub(1);
        std::lock_guard<std::mutex> lk(st->mtx);
        st->seq.push_back(v);
    };
}

// ---------------------------------------------------------------------------
// Section 2 (RED): the callback contract. Values finite and in [0,1], monotonic,
// final exactly 1.0, and never more than one callback in flight.
// ---------------------------------------------------------------------------
void section_progress_contract() {
    ++g_sections;
    const int   R  = 40;
    const float vs = 0.02f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    fillSphere(vol, R, vs, 0.34f);

    auto probe = std::make_shared<ProgressState>();
    MarchingCubes().extract(vol, makeProbe(probe));

    std::lock_guard<std::mutex> lk(probe->mtx);
    size_t non_finite = 0, out_of_range = 0, regressions = 0;
    for (size_t i = 0; i < probe->seq.size(); ++i) {
        const float v = probe->seq[i];
        if (!std::isfinite(v)) ++non_finite;
        if (!(v >= 0.0f && v <= 1.0f)) ++out_of_range;
        if (i > 0 && v < probe->seq[i - 1]) ++regressions;
    }
    const float last = probe->seq.empty() ? -1.0f : probe->seq.back();
    std::printf("s2: calls=%zu max_inflight=%d non_finite=%zu out_of_range=%zu regressions=%zu last=%g\n",
                probe->seq.size(), probe->max_inflight.load(), non_finite, out_of_range, regressions, last);
    CHECK(!probe->seq.empty(), "s2: progress was reported");
    CHECK(non_finite == 0, "s2: every progress value is finite");
    CHECK(out_of_range == 0, "s2: every progress value is within [0, 1]");
    CHECK(regressions == 0, "s2: the progress sequence is nondecreasing");
    CHECK(probe->max_inflight.load() <= 1, "s2: the callback is never invoked concurrently");
    CHECK(probe->seq.back() == 1.0f, "s2: the final progress value is exactly 1.0");
}

// ---------------------------------------------------------------------------
// Section 3 (RED): determinism. Two extractions emit the identical progress
// sequence, independent of OpenMP scheduling.
// ---------------------------------------------------------------------------
void section_progress_determinism() {
    ++g_sections;
    const int   R  = 36;
    const float vs = 0.02f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    fillSphere(vol, R, vs, 0.30f);

    auto a = std::make_shared<ProgressState>();
    auto b = std::make_shared<ProgressState>();
    MarchingCubes().extract(vol, makeProbe(a));
    MarchingCubes().extract(vol, makeProbe(b));

    std::lock_guard<std::mutex> lk(a->mtx);
    std::lock_guard<std::mutex> lk2(b->mtx);
    bool same = (a->seq.size() == b->seq.size());
    if (same)
        for (size_t i = 0; i < a->seq.size(); ++i)
            if (a->seq[i] != b->seq[i]) { same = false; break; }
    std::printf("s3: run1_calls=%zu run2_calls=%zu identical=%d\n",
                a->seq.size(), b->seq.size(), same ? 1 : 0);
    CHECK(!a->seq.empty(), "s3: progress reported in both runs");
    CHECK(same, "s3: the progress sequence is identical across two runs");
}

// ---------------------------------------------------------------------------
// Section 4 (RED): tiny resolutions must not divide by a (resolution - 2) span and
// must never emit a non-finite progress value; a degenerate grid yields no geometry
// rather than crashing.
// ---------------------------------------------------------------------------
void section_tiny_resolution() {
    ++g_sections;

    // Resolution 2: one cube, a single negative corner -> one triangle, but the
    // (resolution - 2) progress span is zero.
    {
        TSDFVolume vol(makeParams(2, 0.1f, 0.3f));
        vol.voxelAt(0, 0, 0).tsdf = -0.5f; vol.voxelAt(0, 0, 0).weight = 8.0f;
        for (int i = 1; i < 2; ++i) {
            vol.voxelAt(i, 0, 0).tsdf = 0.5f; vol.voxelAt(i, 0, 0).weight = 8.0f;
            vol.voxelAt(0, i, 0).tsdf = 0.5f; vol.voxelAt(0, i, 0).weight = 8.0f;
            vol.voxelAt(0, 0, i).tsdf = 0.5f; vol.voxelAt(0, 0, i).weight = 8.0f;
            vol.voxelAt(i, i, 0).tsdf = 0.5f; vol.voxelAt(i, i, 0).weight = 8.0f;
            vol.voxelAt(i, 0, i).tsdf = 0.5f; vol.voxelAt(i, 0, i).weight = 8.0f;
            vol.voxelAt(0, i, i).tsdf = 0.5f; vol.voxelAt(0, i, i).weight = 8.0f;
            vol.voxelAt(i, i, i).tsdf = 0.5f; vol.voxelAt(i, i, i).weight = 8.0f;
        }
        auto probe = std::make_shared<ProgressState>();
        auto m = MarchingCubes().extract(vol, makeProbe(probe));
        std::lock_guard<std::mutex> lk(probe->mtx);
        size_t bad = 0;
        for (float v : probe->seq) if (!std::isfinite(v) || v < 0.0f || v > 1.0f) ++bad;
        const bool geometry_ok = m && (m->empty() || m->validate(nullptr));
        std::printf("s4a: res2 progress_calls=%zu bad=%zu geometry_ok=%d\n",
                    probe->seq.size(), bad, geometry_ok ? 1 : 0);
        CHECK(geometry_ok, "s4a: resolution 2 yields empty or a valid mesh (no crash)");
        CHECK(bad == 0, "s4a: resolution 2 never reports a non-finite/out-of-range progress value");
    }

    // Resolution 1: fewer than one cube; extraction returns nothing without crashing.
    {
        TSDFVolume vol(makeParams(1, 0.1f, 0.3f));
        auto m = MarchingCubes().extract(vol);
        const bool fine = (m != nullptr) && m->empty();
        std::printf("s4b: res1 empty=%d\n", (m && m->empty()) ? 1 : 0);
        CHECK(fine, "s4b: resolution 1 returns an empty mesh without crashing");
    }
}

} // namespace

int main() {
    section_truncation();
    section_progress_contract();
    section_progress_determinism();
    section_tiny_resolution();

    if (g_failures == 0) {
        std::printf("mesh_truncation_progress_contract: PASS (%d sections)\n", g_sections);
        return 0;
    }
    std::printf("mesh_truncation_progress_contract: FAIL (%d failed check(s), %d sections)\n", g_failures, g_sections);
    return 1;
}
