#pragma once

// Qt-only glue between QSurfaceFormat and the Qt-free GlFormatSpec that the
// format ladder reasons about. Kept OUT of gui/GlDiagnostics.h on purpose: the
// decision module must stay Qt-free so it links and runs inside azu_test_core
// with no Qt, no display and no GL. Anything that names QSurfaceFormat belongs
// here, in the display-dependent layer (src/main.cpp, src/gui/OpenGLWidget.cpp)
// that is compiled only into the GUI target.

#include <QSurfaceFormat>

#include "gui/GlDiagnostics.h"

namespace kfusion {
namespace gui {

inline GlFormatSpec toGlFormat(const QSurfaceFormat& fmt) {
    GlFormatSpec spec;
    switch (fmt.renderableType()) {
        case QSurfaceFormat::OpenGL:  spec.renderable = GlRenderableType::DesktopGL; break;
        case QSurfaceFormat::OpenGLES: spec.renderable = GlRenderableType::OpenGLES; break;
        default:                      spec.renderable = GlRenderableType::Unknown; break;
    }
    switch (fmt.profile()) {
        case QSurfaceFormat::CoreProfile:          spec.profile = GlProfile::CoreProfile; break;
        case QSurfaceFormat::CompatibilityProfile: spec.profile = GlProfile::CompatibilityProfile; break;
        case QSurfaceFormat::NoProfile:            spec.profile = GlProfile::NoProfile; break;
        default:                                   spec.profile = GlProfile::Unknown; break;
    }
    spec.major = fmt.majorVersion();
    spec.minor = fmt.minorVersion();
    spec.samples = fmt.samples();
    spec.depth_bits = fmt.depthBufferSize();
    return spec;
}

inline void applyGlFormat(const GlFormatSpec& spec, QSurfaceFormat& fmt) {
    fmt.setRenderableType(spec.renderable == GlRenderableType::OpenGLES
                              ? QSurfaceFormat::OpenGLES
                              : QSurfaceFormat::OpenGL);
    fmt.setProfile(spec.profile == GlProfile::CoreProfile
                       ? QSurfaceFormat::CoreProfile
                       : (spec.profile == GlProfile::CompatibilityProfile
                              ? QSurfaceFormat::CompatibilityProfile
                              : QSurfaceFormat::NoProfile));
    fmt.setVersion(spec.major, spec.minor);
    fmt.setDepthBufferSize(spec.depth_bits);
    // samples() == 0 is the explicit "no MSAA" request (Qt's own default is
    // -1/"unsolicited"); it is how the ladder asks for a non-multisampled config.
    fmt.setSamples(spec.samples > 0 ? spec.samples : 0);
    fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
}

} // namespace gui
} // namespace kfusion
