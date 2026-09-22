#pragma once

// Pure, Qt-free GL-context diagnostics for the GUI launch path.
//
// This header carries NO dependency on Qt, OpenGL, EGL, a display, a device or
// any GPU backend: it is built on the standard library only, so it compiles and
// is unit-tested inside the Qt-free azu_test_core lane (tests/gl_diagnostics_-
// contract). Everything that ACTUALLY touches a QSurfaceFormat, a QOpenGLContext
// or EGL lives in the callers (src/main.cpp, src/gui/OpenGLWidget.cpp) and is
// reduced to the plain structs below before any decision is made here. That
// split is deliberate: the decision logic — which EGL code means what, which
// rung of the format ladder to try next, whether an obtained context clears the
// renderer's hard floor, and how a requested format differs from the one Qt
// actually handed back — is exactly the part that must be provable with no
// display and no GPU, because that is the environment a launch failure is
// reported from.
//
// HARD RENDERER FLOOR (do not "fix" a launch by relaxing this): every preview
// shader is `#version 330 core` (src/rendering/PreviewRenderer.cpp) and
// PreviewRenderer derives from QOpenGLFunctions_3_3_Core. A 330-core shader
// will NOT compile under OpenGL ES or under a sub-3.3 / no-profile context, so
// the only honest ladder is desktop-GL-only and never drops below 3.3 Core. If
// no rung succeeds the caller must fail LOUDLY; silently accepting an ES or 2.x
// context produces a black viewport, which is a worse bug than the crash it
// would have hidden.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kfusion {
namespace gui {

// The renderer's hard floor: desktop OpenGL >= 3.3, Core profile.
inline constexpr int kRequiredGlMajor = 3;
inline constexpr int kRequiredGlMinor = 3;

// Renderable API. Only DesktopGL clears the floor; OpenGLES is recorded so a
// host that handed back an ES context is named in the log rather than mistaken
// for a working desktop one.
enum class GlRenderableType { DesktopGL, OpenGLES, Unknown };

// Context profile. NoProfile / Compatibility are recorded faithfully but only
// CoreProfile clears the floor (the shaders are `core`).
enum class GlProfile { NoProfile, CoreProfile, CompatibilityProfile, Unknown };

// A surface-format request or a created-context description, reduced to the
// fields that decide whether rendering can work. MSAA `samples` is a
// PREFERENCE: a context that clears the floor with samples == 0 is a success,
// because the ladder trades MSAA away before it would ever trade away 3.3 Core.
struct GlFormatSpec {
    GlRenderableType renderable = GlRenderableType::DesktopGL;
    GlProfile        profile     = GlProfile::CoreProfile;
    int major   = kRequiredGlMajor;
    int minor   = kRequiredGlMinor;
    int samples = 0;      // 0 == no MSAA requested
    int depth_bits = 24;
};

// The desktop-GL-only request ladder, highest preference first. Rung 0 keeps the
// original ask (3.3 Core + 4x MSAA); rung 1 drops MSAA, which is the single most
// common cause of an unsatisfiable EGL config request (EGL_BAD_MATCH) and is a
// cosmetic preference, not a correctness requirement. There is deliberately NO
// rung below 3.3 Core: past rung 1 the ladder is exhausted and the caller fails.
const std::vector<GlFormatSpec>& glFormatLadder();

// The next rung to try after `index`, or std::nullopt once the ladder is
// exhausted (which the caller must treat as "fail loudly", never as "try ES").
std::optional<GlFormatSpec> nextFormatRung(std::size_t index);

// True only for a DesktopGL + CoreProfile context of version >= 3.3. Samples and
// depth are intentionally NOT part of the floor.
bool meetsRendererFloor(const GlFormatSpec& actual);

// One-line human-readable form, e.g.
// "DesktopGL 3.3 Core depth=24 samples=4". Used verbatim in the log so a
// requested and an actual line are directly comparable.
std::string describeGlFormat(const GlFormatSpec& spec);

struct GlFormatCompare {
    bool adequate         = false;  // actual clears meetsRendererFloor
    bool matches_request  = false;  // actual renderable+profile+version == requested
    // One truthful line per discrepancy, empty when there is none. A lower
    // sample count is reported as an informational downgrade, not a failure.
    std::vector<std::string> reasons;
};

// Compare what was requested against what the context really is. `adequate`
// answers "can we render"; `matches_request` answers "did we get what we asked
// for" (a rung-1 context is adequate but not a match on samples, which is fine
// and is logged as such).
GlFormatCompare compareGlFormat(const GlFormatSpec& requested, const GlFormatSpec& actual);

// ---- EGL error decoding -------------------------------------------------
// Canonical EGL error codes 0x3001..0x300E, including 0x3003 BAD_ALLOC,
// 0x3005 BAD_CONFIG, 0x3008 BAD_DISPLAY, 0x3009 BAD_MATCH, 0x300D BAD_SURFACE
// and 0x300E CONTEXT_LOST. Anything outside the table maps to
// "EGL_UNKNOWN(0x...)" so an unexpected value is still legible.
std::string eglErrorName(std::uint32_t code);

// An actionable hint for the codes that have a known cause; empty string when
// there is nothing specific to say. For EGL_BAD_MATCH the hint names the two
// real-world producers on this class of host — a desktop profile bit attached to
// an ES context request, and an MSAA / config attribute the EGL config set
// cannot satisfy — and points at the NVIDIA+Mesa dual-vendor override
// __EGL_VENDOR_LIBRARY_FILENAMES as a diagnostic lever.
std::string eglErrorHint(std::uint32_t code);

// "EGL_BAD_MATCH (0x3009): <hint>" — the fully composed single line.
std::string decodeEglError(std::uint32_t code);

// Qt prints EGL failures as free-form text such as
// "QEGLPlatformContext: Failed to create context: 3009". Scan `text` for an
// EGL code token (bare 4-hex-digit like "3009", or an explicit "0x3009") that
// lands in the 0x3000..0x300E range and return the decoded line; std::nullopt
// when no EGL code is present. This is what lets the message handler turn
// Qt's opaque "3009" into an actionable line without the caller parsing it.
std::optional<std::string> decodeEglErrorFromText(const std::string& text);

// Renderable/profile enum to short display strings (stable, log-friendly).
const char* renderableTypeName(GlRenderableType type);
const char* profileName(GlProfile profile);

} // namespace gui
} // namespace kfusion
