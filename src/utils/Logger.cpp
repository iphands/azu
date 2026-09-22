#include "utils/Logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cstdarg>
#include <cstdio>

namespace kfusion {
namespace utils {

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

Logger::Logger() : level_(static_cast<int>(LogLevel::Info)) {}

// Relaxed is sufficient and correct here: the threshold is a standalone scalar
// with no accompanying payload, so no other memory needs to be visible with it.
// What matters is that the read in log()/logf() is a data race free atomic load
// taken OUTSIDE mutex_ (the early-out must not serialize on the emission lock).
void Logger::setLevel(LogLevel level) {
    level_.store(static_cast<int>(level), std::memory_order_relaxed);
}

LogLevel Logger::currentLevel() const {
    return static_cast<LogLevel>(level_.load(std::memory_order_relaxed));
}

void Logger::logf(LogLevel level, const char* tag, const char* format, ...) {
    if (level < currentLevel()) return;

    va_list args;
    va_start(args, format);
    
    // Determine size
    va_list args_copy;
    va_copy(args_copy, args);
    int size = std::vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);

    if (size <= 0) {
        va_end(args);
        return;
    }

    std::vector<char> buffer(size + 1);
    std::vsnprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);

    log(level, tag, std::string(buffer.data()));
}

void Logger::log(LogLevel level, const std::string& tag, const std::string& msg) {
    if (level < currentLevel()) return;

    std::lock_guard<std::mutex> lk(mutex_);

    // Timestamp. localtime_r() writes into THIS call's own std::tm, unlike
    // std::localtime(), which returns a pointer to one process-wide tm that a
    // second logging thread overwrites concurrently (a data race, and a torn
    // hour/minute/second mix in the printed stamp).
    auto now    = std::chrono::system_clock::now();
    auto t      = std::chrono::system_clock::to_time_t(now);
    auto ms     = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;

    char stamp[16] = "00:00:00";
    std::tm local{};
    if (localtime_r(&t, &local) != nullptr) {
        if (std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &local) == 0) {
            std::snprintf(stamp, sizeof(stamp), "00:00:00");
        }
    }
    const int ms_value = static_cast<int>(ms.count()) % 1000;

    const char* lvl_str = "INFO";
    const char* color   = "\033[0m";
    switch (level) {
        case LogLevel::Debug:   lvl_str = "DEBUG"; color = "\033[36m"; break; // Cyan
        case LogLevel::Info:    lvl_str = "INFO "; color = "\033[32m"; break; // Green
        case LogLevel::Warning: lvl_str = "WARN "; color = "\033[33m"; break; // Yellow
        case LogLevel::Error:   lvl_str = "ERROR"; color = "\033[31m"; break; // Red
    }

    // Fixed width tag for alignment (10 chars)
    std::string padded_tag = tag;
    if (padded_tag.length() > 10) padded_tag = padded_tag.substr(0, 7) + "...";
    else while (padded_tag.length() < 10) padded_tag += " ";

    // One composed string, ONE insertion. std::cerr is unbuffered, so every
    // operator<< is its own write(2): a chain of insertions lets another thread
    // writing to std::cerr (or a later logger line) land between the pieces of
    // this line. A single insertion makes each line one indivisible write.
    std::string line;
    line.reserve(64 + msg.size());
    line.append(color);
    line.push_back('[');
    line.append(stamp);
    line.push_back('.');
    line.push_back(static_cast<char>('0' + (ms_value / 100) % 10));
    line.push_back(static_cast<char>('0' + (ms_value / 10) % 10));
    line.push_back(static_cast<char>('0' + ms_value % 10));
    line.append("] [");
    line.append(lvl_str);
    line.append("] [");
    line.append(padded_tag);
    line.append("] ");
    line.append(msg);
    line.append("\033[0m\n");
    std::cerr << line;
}

void Logger::debug(const std::string& tag, const std::string& msg) {
    log(LogLevel::Debug, tag, msg);
}

void Logger::info(const std::string& tag, const std::string& msg) {
    log(LogLevel::Info, tag, msg);
}

void Logger::warn(const std::string& tag, const std::string& msg) {
    log(LogLevel::Warning, tag, msg);
}

void Logger::error(const std::string& tag, const std::string& msg) {
    log(LogLevel::Error, tag, msg);
}

} // namespace utils
} // namespace kfusion
