#include "gui/GlDiagnostics.h"

#include <cctype>
#include <cstdio>
#include <sstream>

namespace kfusion {
namespace gui {

namespace {

// Format a raw code the way the header promises unknown values are shown.
std::string unknownEglName(std::uint32_t code) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "EGL_UNKNOWN(0x%04x)", static_cast<unsigned>(code));
    return std::string(buf);
}

// True when `token` is a bare or 0x-prefixed hex literal whose value is a known
// EGL error code. Qt emits the code as bare hex ("...: 3009"); a caller may also
// hand us an explicit "0x3009". Decimal 3009 is NOT an EGL code, so an unprefixed
// token is interpreted as hex only when it sits in the 0x3000..0x300E band, which
// is exactly the EGL error range and cannot be a coincidental small decimal.
bool parseEglCodeToken(const std::string& token, std::uint32_t* out) {
    if (token.empty()) {
        return false;
    }
    std::string body = token;
    bool explicit_hex = false;
    if (body.size() > 2 && body[0] == '0' && (body[1] == 'x' || body[1] == 'X')) {
        body = body.substr(2);
        explicit_hex = true;
    }
    if (body.empty()) {
        return false;
    }
    for (char c : body) {
        if (std::isxdigit(static_cast<unsigned char>(c)) == 0) {
            return false;
        }
    }
    unsigned long value = 0;
    std::istringstream iss(body);
    iss >> std::hex >> value;
    if (iss.fail() || !iss.eof()) {
        return false;
    }
    const bool in_range = value >= 0x3000UL && value <= 0x300EUL;
    if (!in_range) {
        return false;
    }
    // A bare token that also happens to be a plausible small decimal (e.g. "12")
    // is rejected: only the 0x3000..0x300E band qualifies without an explicit 0x.
    if (!explicit_hex && body.size() != 4) {
        return false;
    }
    *out = static_cast<std::uint32_t>(value);
    return true;
}

} // namespace

const std::vector<GlFormatSpec>& glFormatLadder() {
    // Rung 0: the historical ask, 3.3 Core + 4x MSAA + 24-bit depth.
    // Rung 1: identical but MSAA dropped — the rung that actually launches on a
    // host whose EGL config set cannot satisfy EGL_SAMPLES=4.
    // Nothing else: never below 3.3 Core (see header floor note).
    static const std::vector<GlFormatSpec> kLadder = {
        GlFormatSpec{ GlRenderableType::DesktopGL, GlProfile::CoreProfile,
                      kRequiredGlMajor, kRequiredGlMinor, /*samples=*/4, /*depth=*/24 },
        GlFormatSpec{ GlRenderableType::DesktopGL, GlProfile::CoreProfile,
                      kRequiredGlMajor, kRequiredGlMinor, /*samples=*/0, /*depth=*/24 },
    };
    return kLadder;
}

std::optional<GlFormatSpec> nextFormatRung(std::size_t index) {
    const auto& ladder = glFormatLadder();
    if (index + 1 >= ladder.size()) {
        return std::nullopt;
    }
    return ladder[index + 1];
}

bool meetsRendererFloor(const GlFormatSpec& actual) {
    if (actual.renderable != GlRenderableType::DesktopGL) {
        return false;
    }
    if (actual.profile != GlProfile::CoreProfile) {
        return false;
    }
    if (actual.major < 0 || actual.minor < 0) {
        return false;
    }
    if (actual.major > kRequiredGlMajor) {
        return true;
    }
    if (actual.major < kRequiredGlMajor) {
        return false;
    }
    return actual.minor >= kRequiredGlMinor;
}

const char* renderableTypeName(GlRenderableType type) {
    switch (type) {
        case GlRenderableType::DesktopGL: return "DesktopGL";
        case GlRenderableType::OpenGLES:  return "OpenGLES";
        case GlRenderableType::Unknown:   return "Unknown";
    }
    return "Unknown";
}

const char* profileName(GlProfile profile) {
    switch (profile) {
        case GlProfile::NoProfile:            return "NoProfile";
        case GlProfile::CoreProfile:          return "Core";
        case GlProfile::CompatibilityProfile: return "Compatibility";
        case GlProfile::Unknown:              return "Unknown";
    }
    return "Unknown";
}

std::string describeGlFormat(const GlFormatSpec& spec) {
    std::string out;
    out += renderableTypeName(spec.renderable);
    out += ' ';
    out += std::to_string(spec.major);
    out += '.';
    out += std::to_string(spec.minor);
    out += ' ';
    out += profileName(spec.profile);
    out += " depth=";
    out += std::to_string(spec.depth_bits);
    out += " samples=";
    out += std::to_string(spec.samples);
    return out;
}

GlFormatCompare compareGlFormat(const GlFormatSpec& requested, const GlFormatSpec& actual) {
    GlFormatCompare result;
    result.adequate = meetsRendererFloor(actual);

    const bool same_api = actual.renderable == requested.renderable;
    const bool same_profile = actual.profile == requested.profile;
    const bool same_version = actual.major == requested.major && actual.minor == requested.minor;
    result.matches_request = same_api && same_profile && same_version &&
                             actual.samples >= requested.samples;

    if (!result.adequate) {
        if (actual.renderable == GlRenderableType::OpenGLES) {
            result.reasons.push_back(
                "context is OpenGL ES; the preview shaders are '#version 330 core' and cannot "
                "compile on ES, so rendering would be a black viewport");
        } else if (actual.renderable == GlRenderableType::Unknown) {
            result.reasons.push_back("context renderable type is unknown; cannot confirm desktop GL");
        }
        if (actual.profile != GlProfile::CoreProfile) {
            result.reasons.push_back(std::string("context profile is ") + profileName(actual.profile) +
                                     ", not Core; the shaders require a core profile");
        }
        if (actual.major < kRequiredGlMajor ||
            (actual.major == kRequiredGlMajor && actual.minor < kRequiredGlMinor)) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "context version %d.%d is below the required %d.%d Core floor",
                          actual.major, actual.minor, kRequiredGlMajor, kRequiredGlMinor);
            result.reasons.push_back(buf);
        }
    }

    if (!same_api) {
        result.reasons.push_back(std::string("requested ") + renderableTypeName(requested.renderable) +
                                 " but got " + renderableTypeName(actual.renderable));
    }
    if (!same_profile) {
        result.reasons.push_back(std::string("requested ") + profileName(requested.profile) +
                                 " profile but got " + profileName(actual.profile));
    }
    if (!same_version) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "requested %d.%d but got %d.%d",
                      requested.major, requested.minor, actual.major, actual.minor);
        result.reasons.push_back(buf);
    }
    // MSAA is a preference, so a lower sample count is an informational note and
    // never by itself makes the context inadequate.
    if (actual.samples < requested.samples) {
        char buf[96];
        std::snprintf(buf, sizeof(buf),
                      "MSAA downgraded: requested %d samples, got %d (cosmetic, rendering unaffected)",
                      requested.samples, actual.samples);
        result.reasons.push_back(buf);
    }
    return result;
}

std::string eglErrorName(std::uint32_t code) {
    switch (code) {
        case 0x3000: return "EGL_SUCCESS";
        case 0x3001: return "EGL_NOT_INITIALIZED";
        case 0x3002: return "EGL_BAD_ACCESS";
        case 0x3003: return "EGL_BAD_ALLOC";
        case 0x3004: return "EGL_BAD_ATTRIBUTE";
        case 0x3005: return "EGL_BAD_CONFIG";
        case 0x3006: return "EGL_BAD_CONTEXT";
        case 0x3007: return "EGL_BAD_CURRENT_SURFACE";
        case 0x3008: return "EGL_BAD_DISPLAY";
        case 0x3009: return "EGL_BAD_MATCH";
        case 0x300A: return "EGL_BAD_PARAMETER";
        case 0x300B: return "EGL_BAD_NATIVE_PIXMAP";
        case 0x300C: return "EGL_BAD_NATIVE_WINDOW";
        case 0x300D: return "EGL_BAD_SURFACE";
        case 0x300E: return "EGL_CONTEXT_LOST";
        default:     return unknownEglName(code);
    }
}

std::string eglErrorHint(std::uint32_t code) {
    switch (code) {
        case 0x3009:  // EGL_BAD_MATCH
            return "EGL_BAD_MATCH on this host has two known producers: (1) a desktop "
                   "profile bit attached to an ES context request, and (2) an attribute "
                   "the EGL config set cannot satisfy, above all EGL_SAMPLES=4 (MSAA) when "
                   "no multisample config exists. The app ladder already drops MSAA before "
                   "anything else. With both NVIDIA and Mesa registered as EGL vendors, the "
                   "wrong vendor can be picked; force one with "
                   "__EGL_VENDOR_LIBRARY_FILENAMES (e.g. point it at "
                   "/usr/share/glvnd/egl_vendor.d/50_mesa.json, or 10_nvidia.json) and "
                   "re-run --gl-info to see which vendor answers.";
        case 0x3003:  // EGL_BAD_ALLOC
            return "EGL_BAD_ALLOC: the driver could not allocate the context. Check GPU "
                   "memory pressure and any per-process memory/ulimit caps before retrying.";
        case 0x3005:  // EGL_BAD_CONFIG
            return "EGL_BAD_CONFIG: no EGL config matched the request. On a dual-vendor "
                   "host try __EGL_VENDOR_LIBRARY_FILENAMES to select the other EGL vendor.";
        case 0x3008:  // EGL_BAD_DISPLAY
            return "EGL_BAD_DISPLAY: no usable EGL display. Under a headless/offscreen run "
                   "this is expected; on a session check DISPLAY/WAYLAND_DISPLAY and "
                   "EGL_PLATFORM.";
        case 0x300D:  // EGL_BAD_SURFACE
            return "EGL_BAD_SURFACE: the surface is invalid for the chosen config; the "
                   "window/native surface may be gone or mismatched to the context.";
        case 0x300E:  // EGL_CONTEXT_LOST
            return "EGL_CONTEXT_LOST: the GPU reset or the context was lost (TDR / suspend); "
                   "recreate the context and re-upload GPU resources.";
        default:
            return std::string();
    }
}

std::string decodeEglError(std::uint32_t code) {
    std::string line = eglErrorName(code);
    char hex[16];
    std::snprintf(hex, sizeof(hex), " (0x%04x)", static_cast<unsigned>(code));
    line += hex;
    const std::string hint = eglErrorHint(code);
    if (!hint.empty()) {
        line += ": ";
        line += hint;
    }
    return line;
}

std::optional<std::string> decodeEglErrorFromText(const std::string& text) {
    // Walk the text, split on anything that is not hex-digit or the 0x marker,
    // and test each token. Returns the FIRST decodable EGL code found so the
    // message handler emits exactly one decoded line per Qt message.
    std::string token;
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i <= n) {
        const bool at_end = (i == n);
        const char c = at_end ? '\0' : text[i];
        const bool is_hexish = !at_end &&
            (std::isxdigit(static_cast<unsigned char>(c)) != 0 || c == 'x' || c == 'X');
        if (is_hexish) {
            token.push_back(c);
            ++i;
            continue;
        }
        std::uint32_t code = 0;
        if (parseEglCodeToken(token, &code)) {
            return decodeEglError(code);
        }
        token.clear();
        ++i;
    }
    return std::nullopt;
}

} // namespace gui
} // namespace kfusion
