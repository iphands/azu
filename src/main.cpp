#include <QApplication>
#include <QColor>
#include <QGuiApplication>
#include <QOffscreenSurface>
#include <QMessageBox>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QPalette>
#include <QSurfaceFormat>
#include <QStyleFactory>
#include <QtGlobal>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "sensor/Preprocessor.h"
#include "gui/GlDiagnostics.h"
#include "gui/GlFormatQt.h"
#include "gui/MainWindow.h"
#include "utils/Logger.h"

#ifdef _WIN32
#include <mmsystem.h>
#endif

namespace {

// Exit status for a rejected command line. Distinct from the Qt event-loop
// status so a caller can tell "bad --backend" from "the window closed badly".
constexpr int kUsageErrorExit = 2;

// Exit status when no context clearing the renderer floor could be obtained.
// Distinct from kUsageErrorExit (bad arguments) and from 0 (a clean window
// close), so --gl-info and a refused GUI launch are both machine-detectable.
constexpr int kGlUnavailableExit = 3;

// The build-time GPU backend, read from the target-scoped macros the root
// CMakeLists sets ONLY on the KinectFusionQt target. A CPU build defines
// neither, which is exactly what the CPU lane produces.
constexpr char kBuildBackend[] =
#if defined(CUDA_ENABLED)
    "CUDA";
#elif defined(HIP_ENABLED)
    "HIP";
#else
    "CPU";
#endif

constexpr char kBuildType[] =
#ifdef NDEBUG
    "Release";
#else
    "Debug";
#endif

// QSurfaceFormat <-> GlFormatSpec conversions live in gui/GlFormatQt.h so the
// widget shares the exact same mapping. specFromCurrentContext below is the only
// Qt-GL call left here because it reads the driver's live version string.
kfusion::gui::GlFormatSpec specFromCurrentContext() {
    kfusion::gui::GlFormatSpec spec;
    QOpenGLContext* ctx = QOpenGLContext::currentContext();
    if (!ctx) {
        spec.renderable = kfusion::gui::GlRenderableType::Unknown;
        spec.profile = kfusion::gui::GlProfile::Unknown;
        spec.major = 0;
        spec.minor = 0;
        return spec;
    }
    spec = kfusion::gui::toGlFormat(ctx->format());
    // Trust the driver's own reported version over the requested format.
    QOpenGLFunctions functions;
    functions.initializeOpenGLFunctions();
    const unsigned char* version =
        reinterpret_cast<const unsigned char*>(glGetString(GL_VERSION));
    if (version) {
        int major = 0, minor = 0;
        if (std::sscanf(reinterpret_cast<const char*>(version), "%d.%d", &major, &minor) == 2) {
            spec.major = major;
            spec.minor = minor;
        }
    }
    return spec;
}

std::string glString(GLenum name) {
    QOpenGLContext* ctx = QOpenGLContext::currentContext();
    if (!ctx) return "(no context)";
    const unsigned char* raw = glGetString(name);
    return raw ? std::string(reinterpret_cast<const char*>(raw)) : std::string("(null)");
}

// ---- Qt message handler: route Qt into KFLOG, dedup, decode EGL ------------
// Qt's own EGL/QRhi warnings ("QEGLPlatformContext: Failed to create context:
// 3009") bypass the project logger entirely and, on this failure, the same few
// lines print five times. They are routed into KFLOG (category + file:line
// preserved), an embedded EGL code is decoded to a name + actionable hint, and
// repeats of the same message are emitted ONCE with the total count reported at
// shutdown — so a redirected log shows one legible error, not a wall of noise.
std::mutex g_qt_msg_mutex;
std::map<std::string, int> g_qt_message_counts;
std::vector<std::string> g_qt_message_order;      // first-seen order, for the summary

void emitQtMessageLine(const std::string& line, QtMsgType type) {
    switch (type) {
        case QtDebugMsg:    KFLOG_DEBUG("qt", line); break;
        case QtInfoMsg:     KFLOG_INFO("qt", line); break;
        case QtWarningMsg:  KFLOG_WARN("qt", line); break;
        case QtCriticalMsg:
        case QtFatalMsg:    KFLOG_ERROR("qt", line); break;
    }
}

void qtMessageHandler(QtMsgType type, const QMessageLogContext& context, const QString& message) {
    const std::string text = message.toStdString();
    const std::string category = context.category ? context.category : "default";

    // Preserve Qt's category and source location in the project logger line.
    std::string line = "[" + category;
    if (context.file) {
        line += "@";
        line += context.file;
        if (context.line > 0) {
            line += ":";
            line += std::to_string(context.line);
        }
    }
    line += "] ";
    line += text;

    // A message that carries an EGL error code gets one decoded, actionable
    // suffix so "3009" becomes "EGL_BAD_MATCH (0x3009): <hint>".
    if (const auto decoded = kfusion::gui::decodeEglErrorFromText(text)) {
        line += "  ->  ";
        line += *decoded;
    }

    const QtMsgType captured = type;
    const std::string signature = category + "\n" + text;
    std::lock_guard<std::mutex> lock(g_qt_msg_mutex);
    int& count = g_qt_message_counts[signature];
    if (count == 0) {
        g_qt_message_order.push_back(signature);
        count = 1;
        emitQtMessageLine(line, captured);
    } else {
        ++count;  // already shown once; only tally the repeat
    }
}

// Report how many times each distinct Qt message repeated, so the collapsed
// lines stay honest about the volume of the original burst.
void flushQtMessageSummary() {
    std::lock_guard<std::mutex> lock(g_qt_msg_mutex);
    for (const std::string& sig : g_qt_message_order) {
        const int count = g_qt_message_counts[sig];
        if (count > 1) {
            KFLOG_WARN("qt", "message repeated " + std::to_string(count) +
                                 " times in total (shown once): " + sig.substr(sig.find('\n') + 1));
        }
    }
    g_qt_message_counts.clear();
    g_qt_message_order.clear();
}

// ---- Environment + format report shared by --gl-info and a refused launch ---
const char* envOrUnset(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : "(unset)";
}

// Try every ladder rung off an offscreen surface and return the first that
// yields a context clearing the renderer floor. Emits a report through the
// logger and (when sink != nullptr) to stdout for --gl-info. The return value
// is the winning rung, or nullopt when none can obtain an adequate context.
std::optional<kfusion::gui::GlFormatSpec> probeGlFormats(std::ostream* sink) {
    auto report = [&](const std::string& line) {
        KFLOG_INFO("gl-info", line);
        if (sink) {
            (*sink) << line << "\n";
            sink->flush();
        }
    };

    report(std::string("platform: ") + (QGuiApplication::platformName().isEmpty()
                                            ? "(none)"
                                            : QGuiApplication::platformName().toStdString()));
    report(std::string("Qt runtime: ") + qVersion());
    report(std::string("env DISPLAY=") + envOrUnset("DISPLAY"));
    report(std::string("env WAYLAND_DISPLAY=") + envOrUnset("WAYLAND_DISPLAY"));
    report(std::string("env QT_QPA_PLATFORM=") + envOrUnset("QT_QPA_PLATFORM"));
    report(std::string("env QT_OPENGL=") + envOrUnset("QT_OPENGL"));
    report(std::string("env __EGL_VENDOR_LIBRARY_FILENAMES=") + envOrUnset("__EGL_VENDOR_LIBRARY_FILENAMES"));
    report(std::string("env EGL_PLATFORM=") + envOrUnset("EGL_PLATFORM"));
    report(std::string("env LIBGL_ALWAYS_SOFTWARE=") + envOrUnset("LIBGL_ALWAYS_SOFTWARE"));
    report("default format (pre-probe): " +
           kfusion::gui::describeGlFormat(kfusion::gui::toGlFormat(QSurfaceFormat::defaultFormat())));

    std::optional<kfusion::gui::GlFormatSpec> winner;
    const auto& ladder = kfusion::gui::glFormatLadder();
    for (std::size_t rung = 0; rung < ladder.size(); ++rung) {
        const kfusion::gui::GlFormatSpec& spec = ladder[rung];
        QSurfaceFormat fmt;
        kfusion::gui::applyGlFormat(spec, fmt);

        QOffscreenSurface surface;
        surface.setFormat(fmt);
        surface.create();

        QOpenGLContext ctx;
        ctx.setFormat(fmt);
        const bool created = ctx.create();

        std::string line = "rung " + std::to_string(rung) + " request " +
                           kfusion::gui::describeGlFormat(spec) + " : ";
        // On failure Qt itself prints the decoded EGL code through the installed
        // message handler, so the create-failed line stays terse; the hint is
        // already in the log.
        if (!created || !surface.isValid() || !ctx.makeCurrent(&surface)) {
            line += "context create FAILED";
            report(line);
            continue;
        }

        const kfusion::gui::GlFormatSpec actual = specFromCurrentContext();
        line += "context OK, actual " + kfusion::gui::describeGlFormat(actual);
        report(line);
        report("  GL_VENDOR=" + glString(GL_VENDOR) + " | GL_RENDERER=" + glString(GL_RENDERER) +
               " | GL_VERSION=" + glString(GL_VERSION) +
               " | GL_SHADING_LANGUAGE_VERSION=" + glString(GL_SHADING_LANGUAGE_VERSION));

        const bool adequate = kfusion::gui::meetsRendererFloor(actual);
        if (adequate) {
            report("  adequate: YES (desktop >= 3.3 Core)");
            winner = spec;  // record the REQUEST that worked, to install as default
            ctx.doneCurrent();
            break;
        }
        report("  adequate: NO (does not clear the 3.3 Core desktop floor; continuing the ladder)");
        ctx.doneCurrent();
    }

    if (winner) {
        report("VERDICT: adequate context = YES (rung obtained a desktop >= 3.3 Core context)");
    } else {
        report("VERDICT: adequate context = NO — no ladder rung obtained a desktop >= 3.3 Core "
               "context. The renderer cannot run below that floor (shaders are '#version 330 "
               "core'); with NVIDIA+Mesa both registered, try forcing one EGL vendor via "
               "__EGL_VENDOR_LIBRARY_FILENAMES and re-run --gl-info.");
    }
    return winner;
}

// A real windowing platform (as opposed to offscreen/minimal) is the only case
// where a modal dialog is both visible and non-hanging.
bool platformHasVisibleWindow() {
    const QString name = QGuiApplication::platformName();
    return !name.isEmpty() && name != "offscreen" && name != "minimal";
}

// The literal --backend values this binary accepts, case-insensitively.
// "gpu" is the accepted alias for CUDA and "auto" the default, so the set is
// exactly what kfusion::sensor::parseBackendName() recognizes plus "auto".
// Big-fix Todo 35 (audit pipeline:MN-01): parseBackendName() maps EVERY
// unrecognized string to Auto, so on its own it cannot tell "cpu" from
// "garbage" - the accepted names are matched explicitly here at the call site.
const char* const kAcceptedBackendNames[] = {"auto", "cpu", "cuda", "gpu", "hip"};
constexpr char kAcceptedBackendList[] = "auto, cpu, cuda, gpu, hip";

bool isAcceptedBackendName(const std::string& name) {
    std::string lowered;
    lowered.reserve(name.size());
    for (char c : name) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    for (const char* accepted : kAcceptedBackendNames) {
        if (lowered == accepted) {
            return true;
        }
    }
    return false;
}

} // namespace

// Validate --backend BEFORE QApplication is constructed. That ordering is the
// point, not a style preference: QApplication needs a QPA platform, so an
// argument error reported afterwards is unobservable to a headless caller
// (CI, scripts) that has no display. Policy chosen and documented: hard ERROR
// and exit(kUsageErrorExit) naming the offending value - NOT a warning plus a
// silent fallback to Auto/CPU, which is exactly the audit finding
// (pipeline:MN-01) this closes. A missing value ("--backend" as the last
// argument) is rejected the same way; the old `i + 1 < argc` guard ignored it.
bool validateBackendArgs(int argc, char* argv[], std::string* out_value) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") != 0) {
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "[KinectFusionQt] error: --backend requires a value"
                      << " (accepted: " << kAcceptedBackendList << ")"
                      << "; got no argument after --backend\n";
            return false;
        }
        const std::string value = argv[++i];
        if (!isAcceptedBackendName(value)) {
            std::cerr << "[KinectFusionQt] error: unrecognized --backend value '" << value
                      << "' (accepted: " << kAcceptedBackendList << ")"
                      << "; note cuda/gpu and hip are accepted but DEFERRED backends in this"
                      << " build and resolve to CPU at runtime\n";
            return false;
        }
        if (out_value) {
            *out_value = value;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    timeBeginPeriod(1);
#endif

    // CLI validation stays FIRST, before any Qt platform work: a rejected
    // --backend must be observable to a headless caller (see the invariant
    // comment above validateBackendArgs). This is the pipeline:MN-01 contract.
    std::string backend_arg;
    if (!validateBackendArgs(argc, argv, &backend_arg)) {
        return kUsageErrorExit;
    }

    // Verbose and --gl-info are pure argv/env reads, done before QApplication so
    // the logger threshold and the startup trace exist even if platform init
    // aborts. parseBackendName() is reached only with an already-accepted value.
    bool verbose_cli = false;
    bool gl_info = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0) {
            verbose_cli = true;
        } else if (std::strcmp(argv[i], "--gl-info") == 0) {
            gl_info = true;
        }
    }
    const char* env_log = std::getenv("KFUSION_LOG");
    std::string env_str = env_log ? env_log : "";
    for (auto& c : env_str) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (verbose_cli || env_str == "debug" || env_str == "1" || env_str == "verbose") {
        kfusion::utils::Logger::instance().setLevel(kfusion::utils::LogLevel::Debug);
        std::cerr << "[KinectFusionQt] Verbose logging: KFUSION_LOG=debug or --verbose\n";
        std::cerr.flush();
    }

    // Route Qt's EGL/QRhi warnings into KFLOG and decode EGL codes BEFORE any
    // context is attempted, so the very first failure is captured and legible.
    qInstallMessageHandler(qtMessageHandler);

    const kfusion::sensor::PreprocessBackend preferred_backend =
        backend_arg.empty()
            ? kfusion::sensor::PreprocessBackend::Auto
            : kfusion::sensor::parseBackendName(backend_arg);

    // Startup banner EARLY and flushed. The old trace printed only after
    // QApplication existed, through buffered std::cout that a pre-abort discarded
    // — so a redirected run looked empty. This goes through the logger AND an
    // explicit flush, and names the build/backend right before display access.
    KFLOG_INFO("app",
               std::string("KinectFusionQt v1.0 starting (build=") + kBuildType +
                   " backend=" + kBuildBackend + "); backend preference=" +
                   kfusion::sensor::backendName(preferred_backend) +
                   "; about to initialise the Qt platform and acquire a GL context");
    std::cout << "KinectFusionQt v1.0 starting...\n";
    std::cout << "Preprocess backend preference: "
              << kfusion::sensor::backendName(preferred_backend) << "\n";
    std::cout.flush();

    // Install the preferred desktop 3.3 Core format (ladder rung 0, MSAA kept)
    // globally before QApplication, exactly as before; the probe below may later
    // downgrade the default to rung 1 (no MSAA) if that is all the host can give.
    {
        QSurfaceFormat fmt;
        kfusion::gui::applyGlFormat(kfusion::gui::glFormatLadder()[0], fmt);
        QSurfaceFormat::setDefaultFormat(fmt);
    }

    // Qt5 line removed for the Qt6 port: Qt::AA_EnableHighDpiScaling was
    // removed from Qt — high-Dpi scaling is always enabled in Qt 6.

    QApplication app(argc, argv);
    app.setApplicationName("KinectFusionQt");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("KinectFusion");

    if (gl_info) {
        // Headless-safe diagnostics: full environment + per-rung context report,
        // then exit. 0 when an adequate context was obtained, non-zero otherwise.
        const auto winner = probeGlFormats(&std::cout);
        flushQtMessageSummary();
#ifdef _WIN32
        timeEndPeriod(1);
#endif
        return winner ? 0 : kGlUnavailableExit;
    }

    // Pick the format the window will actually use by trying the ladder for real.
    // On success the winning rung becomes the default so the widget inherits it;
    // MSAA is dropped before anything else, never 3.3 Core. On total failure the
    // app refuses to open a black window and fails LOUDLY.
    const auto winner = probeGlFormats(nullptr);
    if (!winner) {
        const std::string msg =
            "No OpenGL context clearing the desktop 3.3 Core floor could be obtained. "
            "The preview shaders are '#version 330 core' and the renderer needs 3.3 Core, "
            "so opening a window would show a black viewport. Run with --gl-info for the "
            "full report; with NVIDIA and Mesa both registered as EGL vendors, try forcing "
            "one vendor via __EGL_VENDOR_LIBRARY_FILENAMES and retry.";
        KFLOG_ERROR("gl", msg);
        std::cerr << "[KinectFusionQt] ERROR: " << msg << "\n";
        std::cerr.flush();
        if (platformHasVisibleWindow()) {
            QMessageBox::critical(nullptr, "KinectFusionQt: OpenGL context unavailable",
                                  QString::fromStdString(msg));
        }
        flushQtMessageSummary();
#ifdef _WIN32
        timeEndPeriod(1);
#endif
        return kGlUnavailableExit;
    }

    QSurfaceFormat chosen;
    kfusion::gui::applyGlFormat(*winner, chosen);
    QSurfaceFormat::setDefaultFormat(chosen);
    KFLOG_INFO("gl", "default format set to: " + kfusion::gui::describeGlFormat(*winner));

    // Apply Fusion dark style
    app.setStyle(QStyleFactory::create("Fusion"));
    QPalette dark;
    dark.setColor(QPalette::Window,          QColor(30, 31, 36));
    dark.setColor(QPalette::WindowText,      QColor(220, 220, 220));
    dark.setColor(QPalette::Base,            QColor(22, 23, 28));
    dark.setColor(QPalette::AlternateBase,   QColor(35, 36, 42));
    dark.setColor(QPalette::Text,            QColor(220, 220, 220));
    dark.setColor(QPalette::Button,          QColor(45, 46, 54));
    dark.setColor(QPalette::ButtonText,      QColor(220, 220, 220));
    dark.setColor(QPalette::Highlight,       QColor(58, 127, 207));
    dark.setColor(QPalette::HighlightedText, QColor(255, 255, 255));
    dark.setColor(QPalette::ToolTipBase,     QColor(50, 50, 60));
    dark.setColor(QPalette::ToolTipText,     QColor(200, 200, 200));
    app.setPalette(dark);

    kfusion::gui::MainWindow window(preferred_backend);
    window.show();
    const int exit_code = app.exec();

    flushQtMessageSummary();

#ifdef _WIN32
    timeEndPeriod(1);
#endif

    return exit_code;
}
