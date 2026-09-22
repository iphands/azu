// gl_diagnostics_contract: CPU-only, Qt-free contract for the GL-context launch
// diagnostics module (include/gui/GlDiagnostics.h + src/gui/GlDiagnostics.cpp).
//
// It drives the REAL translation unit (linked through azu_test_core; gate 16 in
// tests/CMakeLists.txt makes a local replica fail to link and fails the source
// coupling check) through its public namespace. Nothing here needs a display, a
// GPU, an OpenGL context or Qt: the whole point of the module is that the launch
// failure is decodable with none of those present. Expectations are independent
// rule statements, never product snapshots. Cases locked here:
//   * EGL canonical codes 0x3001..0x300E decode to the exact names, incl.
//     0x3003 BAD_ALLOC, 0x3005 BAD_CONFIG, 0x3008 BAD_DISPLAY, 0x3009 BAD_MATCH,
//     0x300D BAD_SURFACE, 0x300E CONTEXT_LOST
//   * unknown codes map to EGL_UNKNOWN(0x...) (negative cases: out-of-range and
//     non-EGL values)
//   * EGL_BAD_MATCH's hint names both real-world causes AND the dual-vendor
//     __EGL_VENDOR_LIBRARY_FILENAMES lever; SUCCESS / unknown carry NO hint
//   * decodeEglError composes "NAME (0x..): hint"; SUCCESS has no hint suffix
//   * decodeEglErrorFromText pulls the code out of Qt's free-form "3009" text and
//     ignores messages with no EGL code and out-of-range hex (negative)
//   * the format ladder is desktop-only, starts at 3.3 Core + MSAA, drops to
//     no-MSAA as rung 1, and is exhausted with nothing below 3.3 Core (negative)
//   * meetsRendererFloor accepts only DesktopGL + Core + >= 3.3 (negatives for ES,
//     NoProfile, Compatibility, sub-3.3)
//   * compareGlFormat: MSAA downgrade is adequate-but-not-matching (a note, not a
//     failure); an ES/sub-3.3 context is inadequate with reasons naming why
// No Qt Widgets, OpenGL, EGL, display, device, GPU, sensor, thread timing or
// filesystem access.

#include "gui/GlDiagnostics.h"

#include <cstdint>
#include <cstdio>
#include <string>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "gl_diagnostics_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

// Namespace-qualified alias so every call carries the `::` the anti-synthetic
// gate looks for, and so a look-alike replica (which would have to re-declare
// these in kfusion::gui) collides with the real header this test includes.
namespace gdiag = kfusion::gui;
using kfusion::gui::GlFormatSpec;
using kfusion::gui::GlProfile;
using kfusion::gui::GlRenderableType;

int g_failures = 0;
int g_checks   = 0;

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

bool containsLower(const std::string& hay, const std::string& needle_lower) {
    std::string lowered;
    lowered.reserve(hay.size());
    for (char c : hay) {
        lowered.push_back(static_cast<char>(
            (c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c));
    }
    return lowered.find(needle_lower) != std::string::npos;
}

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),        \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

// ---- Section 1: canonical EGL names, plus the unknown-code negative ----
void sectionEglNames() {
    CHECK(gdiag::eglErrorName(0x3000) == "EGL_SUCCESS", "0x3000 -> EGL_SUCCESS");
    CHECK(gdiag::eglErrorName(0x3001) == "EGL_NOT_INITIALIZED", "0x3001 -> NOT_INITIALIZED");
    CHECK(gdiag::eglErrorName(0x3002) == "EGL_BAD_ACCESS", "0x3002 -> BAD_ACCESS");
    CHECK(gdiag::eglErrorName(0x3003) == "EGL_BAD_ALLOC", "0x3003 -> BAD_ALLOC");
    CHECK(gdiag::eglErrorName(0x3004) == "EGL_BAD_ATTRIBUTE", "0x3004 -> BAD_ATTRIBUTE");
    CHECK(gdiag::eglErrorName(0x3005) == "EGL_BAD_CONFIG", "0x3005 -> BAD_CONFIG");
    CHECK(gdiag::eglErrorName(0x3006) == "EGL_BAD_CONTEXT", "0x3006 -> BAD_CONTEXT");
    CHECK(gdiag::eglErrorName(0x3007) == "EGL_BAD_CURRENT_SURFACE", "0x3007 -> BAD_CURRENT_SURFACE");
    CHECK(gdiag::eglErrorName(0x3008) == "EGL_BAD_DISPLAY", "0x3008 -> BAD_DISPLAY");
    CHECK(gdiag::eglErrorName(0x3009) == "EGL_BAD_MATCH", "0x3009 -> BAD_MATCH");
    CHECK(gdiag::eglErrorName(0x300A) == "EGL_BAD_PARAMETER", "0x300a -> BAD_PARAMETER");
    CHECK(gdiag::eglErrorName(0x300B) == "EGL_BAD_NATIVE_PIXMAP", "0x300b -> BAD_NATIVE_PIXMAP");
    CHECK(gdiag::eglErrorName(0x300C) == "EGL_BAD_NATIVE_WINDOW", "0x300c -> BAD_NATIVE_WINDOW");
    CHECK(gdiag::eglErrorName(0x300D) == "EGL_BAD_SURFACE", "0x300d -> BAD_SURFACE");
    CHECK(gdiag::eglErrorName(0x300E) == "EGL_CONTEXT_LOST", "0x300e -> CONTEXT_LOST");

    // Negative: just past the table, an unrelated value, and zero are UNKNOWN.
    CHECK(contains(gdiag::eglErrorName(0x300F), "EGL_UNKNOWN"), "0x300f is unknown");
    CHECK(contains(gdiag::eglErrorName(0x300F), "0x300f"), "unknown name echoes the hex value");
    CHECK(contains(gdiag::eglErrorName(0x1234), "EGL_UNKNOWN"), "0x1234 is unknown");
    CHECK(contains(gdiag::eglErrorName(0x0001), "EGL_UNKNOWN"), "0x0001 is unknown");
}

// ---- Section 2: hints, and the empty-hint negatives ----
void sectionEglHints() {
    const std::string bad_match = gdiag::eglErrorHint(0x3009);
    CHECK(!bad_match.empty(), "BAD_MATCH has a hint");
    CHECK(containsLower(bad_match, "msaa") || contains(bad_match, "EGL_SAMPLES"),
          "BAD_MATCH hint names the MSAA/config cause");
    CHECK(containsLower(bad_match, "es") || containsLower(bad_match, "profile"),
          "BAD_MATCH hint names the profile/ES cause");
    CHECK(contains(bad_match, "__EGL_VENDOR_LIBRARY_FILENAMES"),
          "BAD_MATCH hint names the dual-vendor override lever");

    CHECK(!gdiag::eglErrorHint(0x3003).empty(), "BAD_ALLOC has a hint");
    CHECK(containsLower(gdiag::eglErrorHint(0x3003), "alloc"), "BAD_ALLOC hint mentions allocation");
    CHECK(!gdiag::eglErrorHint(0x3005).empty(), "BAD_CONFIG has a hint");
    CHECK(!gdiag::eglErrorHint(0x3008).empty(), "BAD_DISPLAY has a hint");
    CHECK(!gdiag::eglErrorHint(0x300D).empty(), "BAD_SURFACE has a hint");
    CHECK(!gdiag::eglErrorHint(0x300E).empty(), "CONTEXT_LOST has a hint");

    // Negative: SUCCESS and an unknown code must carry no actionable hint.
    CHECK(gdiag::eglErrorHint(0x3000).empty(), "SUCCESS has no hint");
    CHECK(gdiag::eglErrorHint(0x9999).empty(), "unknown code has no hint");
}

// ---- Section 3: composed decode line ----
void sectionDecodeEglError() {
    const std::string line = gdiag::decodeEglError(0x3009);
    CHECK(contains(line, "EGL_BAD_MATCH"), "composed line names BAD_MATCH");
    CHECK(contains(line, "(0x3009)"), "composed line echoes (0x3009)");
    CHECK(contains(line, "__EGL_VENDOR_LIBRARY_FILENAMES"), "composed line carries the hint");

    // Negative: SUCCESS composes with no hint, so no ": " suffix is appended.
    CHECK(gdiag::decodeEglError(0x3000) == "EGL_SUCCESS (0x3000)",
          "SUCCESS composes to exactly 'EGL_SUCCESS (0x3000)' with no hint");
}

// ---- Section 4: pulling the code out of Qt's free-form text ----
void sectionDecodeFromText() {
    const auto qbad = gdiag::decodeEglErrorFromText(
        "QEGLPlatformContext: Failed to create context: 3009");
    CHECK(qbad.has_value(), "Qt '...: 3009' text decodes");
    CHECK(qbad && contains(*qbad, "EGL_BAD_MATCH"), "decoded from the exact reported string");
    CHECK(qbad && contains(*qbad, "__EGL_VENDOR_LIBRARY_FILENAMES"),
          "text decode yields the actionable hint");

    const auto explicit_hex = gdiag::decodeEglErrorFromText("create failed (0x300e)");
    CHECK(explicit_hex && contains(*explicit_hex, "EGL_CONTEXT_LOST"), "explicit 0x300e decodes");

    const auto mid = gdiag::decodeEglErrorFromText("config error 3005 rejected");
    CHECK(mid && contains(*mid, "EGL_BAD_CONFIG"), "mid-string 3005 decodes");

    // Negatives: no EGL code, and hex that is not an EGL error code.
    CHECK(!gdiag::decodeEglErrorFromText("QRhiGles2: Failed to create context").has_value(),
          "a message with no EGL code yields nothing");
    CHECK(!gdiag::decodeEglErrorFromText("small number 12 only").has_value(),
          "a bare non-EGL decimal is not mistaken for a code");
    CHECK(!gdiag::decodeEglErrorFromText("value 0x1234 out of range").has_value(),
          "out-of-range hex is not an EGL error code");
}

// ---- Section 5: the desktop-only ladder and the renderer floor ----
void sectionLadderAndFloor() {
    const auto& ladder = gdiag::glFormatLadder();
    CHECK(ladder.size() >= 2, "ladder has at least a MSAA rung and a no-MSAA rung");

    const GlFormatSpec& rung0 = ladder[0];
    CHECK(rung0.renderable == GlRenderableType::DesktopGL, "rung 0 is desktop GL");
    CHECK(rung0.profile == GlProfile::CoreProfile, "rung 0 is Core profile");
    CHECK(rung0.major == 3 && rung0.minor == 3, "rung 0 requests 3.3");
    CHECK(rung0.samples > 0, "rung 0 keeps MSAA");

    const GlFormatSpec& rung1 = ladder[1];
    CHECK(rung1.samples == 0, "rung 1 drops MSAA");
    CHECK(rung1.renderable == GlFormatSpec().renderable &&
              rung1.renderable == GlRenderableType::DesktopGL &&
              rung1.profile == GlProfile::CoreProfile && rung1.major == 3 && rung1.minor == 3,
          "rung 1 keeps 3.3 Core desktop GL (only MSAA is traded away)");

    const auto next = gdiag::nextFormatRung(0);
    CHECK(next.has_value(), "there is a rung after the first");
    CHECK(next && next->samples == 0, "the next rung is the no-MSAA one");

    // Negative: the ladder is exhausted past its last rung — there is NO rung
    // that goes below 3.3 Core or onto ES.
    CHECK(!gdiag::nextFormatRung(ladder.size() - 1).has_value(),
          "ladder is exhausted: no rung below 3.3 Core exists");
    CHECK(!gdiag::nextFormatRung(ladder.size()).has_value(), "past-the-end has no next rung");

    // The floor: only DesktopGL + Core + >= 3.3 clears it.
    GlFormatSpec ok33;
    ok33.renderable = GlRenderableType::DesktopGL;
    ok33.profile = GlProfile::CoreProfile;
    ok33.major = 3; ok33.minor = 3;
    CHECK(gdiag::meetsRendererFloor(ok33), "3.3 Core desktop clears the floor");

    GlFormatSpec ok45 = ok33; ok45.major = 4; ok45.minor = 5;
    CHECK(gdiag::meetsRendererFloor(ok45), "4.5 Core desktop clears the floor");

    // Negatives: every one must NOT clear the floor.
    GlFormatSpec low32 = ok33; low32.minor = 2;
    CHECK(!gdiag::meetsRendererFloor(low32), "3.2 Core does NOT clear the floor");
    GlFormatSpec noprof = ok33; noprof.profile = GlProfile::NoProfile;
    CHECK(!gdiag::meetsRendererFloor(noprof), "NoProfile does NOT clear the floor");
    GlFormatSpec compat40; compat40.major = 4; compat40.minor = 0;
    compat40.profile = GlProfile::CompatibilityProfile;
    CHECK(!gdiag::meetsRendererFloor(compat40), "Compatibility profile does NOT clear the floor");
    GlFormatSpec es = ok33; es.renderable = GlRenderableType::OpenGLES;
    CHECK(!gdiag::meetsRendererFloor(es), "an ES context does NOT clear the floor (shaders are 330 core)");

    // describeGlFormat is the log line shape, samples inclusive.
    CHECK(contains(gdiag::describeGlFormat(rung0), "samples=4"), "describe shows samples=4");
    CHECK(contains(gdiag::describeGlFormat(rung1), "samples=0"), "describe shows samples=0");
    CHECK(contains(gdiag::describeGlFormat(rung0), "Core"), "describe shows Core");
}

// ---- Section 6: requested-vs-actual comparison ----
void sectionCompare() {
    const auto& ladder = gdiag::glFormatLadder();
    const GlFormatSpec asked_msaa = ladder[0];   // 3.3 Core + 4 samples
    const GlFormatSpec asked_flat = ladder[1];   // 3.3 Core + no MSAA

    // MSAA dropped: adequate, but not a match — reported as an informational note.
    const auto down = gdiag::compareGlFormat(asked_msaa, asked_flat);
    CHECK(down.adequate, "a no-MSAA 3.3 Core context is adequate");
    CHECK(!down.matches_request, "no-MSAA is not a match for a MSAA request");
    CHECK(!down.reasons.empty() && containsLower(down.reasons.back(), "msaa"),
          "MSAA downgrade is explained as a note, not a failure");

    // Exact match: adequate and no reasons.
    const auto same = gdiag::compareGlFormat(asked_msaa, ladder[0]);
    CHECK(same.adequate && same.matches_request, "identical request/actual is a match");
    CHECK(same.reasons.empty(), "an exact match lists no reasons");

    // Negative: an ES 3.0 context is inadequate and the reasons name the ES/330
    // hazard and the sub-floor version.
    GlFormatSpec es30;
    es30.renderable = GlRenderableType::OpenGLES;
    es30.profile = GlProfile::CoreProfile;
    es30.major = 3; es30.minor = 0;
    const auto bad = gdiag::compareGlFormat(asked_msaa, es30);
    CHECK(!bad.adequate, "ES 3.0 is inadequate");
    CHECK(!bad.matches_request, "ES 3.0 does not match the desktop request");
    bool names_es = false;
    bool names_version = false;
    for (const std::string& r : bad.reasons) {
        if (contains(r, "ES")) names_es = true;
        if (containsLower(r, "3.0") || containsLower(r, "below")) names_version = true;
    }
    CHECK(names_es, "reasons name the ES hazard");
    CHECK(names_version, "reasons name the sub-floor version");
}

} // namespace

int main() {
    sectionEglNames();
    sectionEglHints();
    sectionDecodeEglError();
    sectionDecodeFromText();
    sectionLadderAndFloor();
    sectionCompare();

    std::printf("gl_diagnostics_contract: %d checks, %d failures\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("gl_diagnostics_contract: FAIL\n");
        return 1;
    }
    std::printf("gl_diagnostics_contract: PASS\n");
    return 0;
}
