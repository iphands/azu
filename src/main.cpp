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
    kfusion::sensor::PreprocessBackend preferred_backend = kfusion::sensor::PreprocessBackend::Auto;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0)
            verbose_cli = true;
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            preferred_backend = kfusion::sensor::parseBackendName(argv[++i]);
        }
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
