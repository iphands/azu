#include <QApplication>
#include <QColor>
#include <QPalette>
#include <QSurfaceFormat>
#include <QStyleFactory>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include "sensor/Preprocessor.h"
#include "gui/MainWindow.h"
#include "utils/Logger.h"

#ifdef _WIN32
#include <mmsystem.h>
#endif

namespace {

// Exit status for a rejected command line. Distinct from the Qt event-loop
// status so a caller can tell "bad --backend" from "the window closed badly".
constexpr int kUsageErrorExit = 2;

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

    // Set OpenGL surface format globally before creating QApplication
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    fmt.setSamples(4);
    fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    QSurfaceFormat::setDefaultFormat(fmt);

    // Qt5 line removed for the Qt6 port: Qt::AA_EnableHighDpiScaling was
    // removed from Qt — high-DPI scaling is always enabled in Qt 6.

    std::string backend_arg;
    if (!validateBackendArgs(argc, argv, &backend_arg)) {
        return kUsageErrorExit;
    }

    QApplication app(argc, argv);
    app.setApplicationName("KinectFusionQt");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("KinectFusion");

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

    bool verbose_cli = false;
    // parseBackendName() is reached only with a value already accepted above.
    kfusion::sensor::PreprocessBackend preferred_backend =
        backend_arg.empty()
            ? kfusion::sensor::PreprocessBackend::Auto
            : kfusion::sensor::parseBackendName(backend_arg);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0)
            verbose_cli = true;
    }
    const char* env_log = std::getenv("KFUSION_LOG");
    std::string env_str = env_log ? env_log : "";
    for (auto& c : env_str) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (verbose_cli || env_str == "debug" || env_str == "1" || env_str == "verbose") {
        kfusion::utils::Logger::instance().setLevel(kfusion::utils::LogLevel::Debug);
        std::cerr << "[KinectFusionQt] Verbose logging: KFUSION_LOG=debug or --verbose\n";
    }

    std::cout << "KinectFusionQt v1.0 starting...\n";
    std::cout << "Preprocess backend preference: " << kfusion::sensor::backendName(preferred_backend) << "\n";

    kfusion::gui::MainWindow window(preferred_backend);
    window.show();
    const int exit_code = app.exec();

#ifdef _WIN32
    timeEndPeriod(1);
#endif

    return exit_code;
}
