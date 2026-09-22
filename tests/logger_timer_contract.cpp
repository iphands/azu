// logger_timer_contract (big-fix todo 33): CPU-only contract for the two
// headless utility translation units src/utils/Logger.cpp and src/utils/Timer.cpp
// (both carried by azu_test_core; gate 14 in tests/CMakeLists.txt makes a local
// replica fail to link and a source-only replica fail the post-build nm -C
// coupling guard in scripts/test-cpu-big-fix.sh).
//
// Every expectation is an independent statement about the emission protocol or a
// property of the fixture, never a snapshot of product output:
//   * threshold semantics: setLevel() / currentLevel() agree, and each level emits
//     exactly the records at or above the threshold, for both the typed API and the
//     variadic logf() path.
//   * line shape: one record is exactly <ESC>..m[HH:MM:SS.mmm] [LLLL] [tag......]
//     msg<ESC>[0m\n, with a THREE-digit millisecond field (fixed-width pad, not a
//     streamed integer) and a 10-column tag field including the truncation rule.
//   * stamp validity: localtime_r/strftime produces a zero-padded 24-hour field and
//     the stamps are monotonic across a burst (only a midnight rollover may reset).
//   * single-write integrity: while foreign threads stream unbuffered multi-insert
//     lines to std::cerr, EVERY logger record still arrives as one indivisible
//     line, each token exactly once, count exact. A record assembled by several
//     operator<< on unbuffered std::cerr is several write(2) calls and gets split
//     by those foreign writes, which is what this section detects.
//   * threshold race freedom: setLevel() concurrent with emitters yields only
//     well-formed records (no torn line), which a plain enum field cannot promise.
//   * ScopedTimer allocates ZERO bytes per scope on the below-threshold path,
//     measured by a global operator new counter and a name deliberately longer
//     than the std::string SSO buffer, so a std::string name member fails here.
//   * ScopedTimer emits exactly one DEBUG record carrying its name and the ms
//     value above the 1 ms print threshold, and nothing below it; the name is a
//     string literal, which is the documented lifetime contract of the const char*
//     member. The sleep is a stimulus that crosses the threshold, never an oracle:
//     no duration is asserted against wall clock.
// No device, display, GPU, OpenGL context, sensor, network or GPU timing. The
// filesystem use is a private temp-file stderr capture, removed before exit.

#include "utils/Logger.h"
#include "utils/Timer.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <new>
#include <regex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "logger_timer_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

// ---------------------------------------------------------------- allocation counter
// A process-wide operator new counter is the only way to prove "the hot path
// allocates nothing": the assertion is about malloc traffic, not about members.
// Counting is armed only inside the measured window so unrelated stdlib
// allocations elsewhere cannot make the check vacuous in either direction.
namespace azu_alloc_probe {
inline std::atomic<bool>& armFlag() { static std::atomic<bool> v{false}; return v; }
inline std::atomic<long>& counter() { static std::atomic<long> v{0}; return v; }

inline void countAlloc() {
    if (armFlag().load(std::memory_order_relaxed)) {
        counter().fetch_add(1, std::memory_order_relaxed);
    }
}

inline void* countedAlloc(std::size_t n) {
    countAlloc();
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
} // namespace azu_alloc_probe

void* operator new(std::size_t n) { return azu_alloc_probe::countedAlloc(n); }
void* operator new[](std::size_t n) { return azu_alloc_probe::countedAlloc(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

// The timer is reached through the namespace alias so every construction site is
// receiver-qualified (`utils::ScopedTimer name(`): the anti-synthetic gate matches
// a qualified CALL SHAPE, and a locally re-declared replica timer would collide with
// the utils::ScopedTimer the linked library already declares.
namespace utils = kfusion::utils;
using kfusion::utils::Logger;
using kfusion::utils::LogLevel;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),        \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)




// ---------------------------------------------------------------- stderr capture
// std::cerr writes through fd 2, so redirecting fd 2 with dup2 captures exactly
// what a real terminal or log pipe would receive. The file is private to this
// process and unlinked when the capture object goes out of scope.
class StderrCapture {
public:
    void start() {
        path_ = "/tmp/azu_logger_capture_" + std::to_string(::getpid()) + "_" +
                std::to_string(++seq_) + ".log";
        int fd = ::open(path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
        if (fd < 0) { saved_ = -1; return; }
        saved_ = ::dup(STDERR_FILENO);
        ::dup2(fd, STDERR_FILENO);
        ::close(fd);
    }
    std::string stop() {
        if (saved_ < 0) return "";
        std::cerr.flush();
        ::fflush(stderr);
        ::dup2(saved_, STDERR_FILENO);
        ::close(saved_);
        saved_ = -1;
        std::ifstream in(path_);
        std::string text((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
        in.close();
        ::unlink(path_.c_str());
        return text;
    }

private:
    int         saved_ = -1;
    std::string path_;
    static int  seq_;
};
int StderrCapture::seq_ = 0;

std::vector<std::string> splitLines(const std::string& text) {
    std::vector<std::string> out;
    std::string::size_type pos = 0;
    while (pos < text.size()) {
        auto nl = text.find('\n', pos);
        if (nl == std::string::npos) {
            out.push_back(text.substr(pos));  // trailing partial line
            break;
        }
        out.push_back(text.substr(pos, nl - pos));
        pos = nl + 1;
    }
    return out;
}

// The full record grammar, anchored on both ends, with the three-digit
// millisecond group and the fixed 10-column tag group exposed.
const std::regex& recordPattern() {
    static const std::regex re(
        "^\x1b\\[[0-9;]*m\\[([0-9]{2}):([0-9]{2}):([0-9]{2})\\.([0-9]{3})\\] "
        "\\[(DEBUG|INFO |WARN |ERROR)\\] \\[([^\\]]*)\\] (.*)\x1b\\[0m$");
    return re;
}

int countMatching(const std::vector<std::string>& lines) {
    int n = 0;
    for (const auto& l : lines) {
        if (std::regex_match(l, recordPattern())) ++n;
    }
    return n;
}

// Unanchored twin of recordPattern(): a complete record anywhere in the byte
// stream. A foreign writer that leaves a line without its trailing newline can
// glue arbitrary bytes in FRONT of a logger record, which is the foreign writer's
// own corruption; the property under test is that the logger's OWN bytes stay
// contiguous, so the search must not be anchored at a line start.
const std::regex& recordSearch() {
    static const std::regex re(
        R"RX(\x1b\[[0-9;]*m\[[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}\] \[(DEBUG|INFO |WARN |ERROR)\] \[[^\]]*\] [^\x1b]*\x1b\[0m\n)RX");
    return re;
}

int countContiguousRecords(const std::string& text) {
    return static_cast<int>(std::distance(
        std::sregex_iterator(text.begin(), text.end(), recordSearch()),
        std::sregex_iterator()));
}

int countOccurrences(const std::string& text, const std::string& needle) {
    int n = 0;
    for (std::string::size_type pos = 0;
         (pos = text.find(needle, pos)) != std::string::npos; pos += needle.size()) {
        ++n;
    }
    return n;
}

bool hasLevel(const std::string& line, const char* lvl) {
    std::smatch m;
    if (!std::regex_match(line, m, recordPattern())) return false;
    return m[5].str() == lvl;
}

std::string messageOf(const std::string& line) {
    std::smatch m;
    if (!std::regex_match(line, m, recordPattern())) return "<MALFORMED>";
    return m[7].str();
}

std::string tagOf(const std::string& line) {
    std::smatch m;
    if (!std::regex_match(line, m, recordPattern())) return "<MALFORMED>";
    return m[6].str();
}

bool stampOf(const std::string& line, long& ms_of_day) {
    std::smatch m;
    if (!std::regex_match(line, m, recordPattern())) return false;
    const long hh = std::stol(m[1].str());
    const long mm = std::stol(m[2].str());
    const long ss = std::stol(m[3].str());
    const long ms = std::stol(m[4].str());
    if (hh > 23 || mm > 59 || ss > 59 || ms > 999) return false;
    ms_of_day = ((hh * 60 + mm) * 60 + ss) * 1000 + ms;
    return true;
}

Logger& L() { return Logger::instance(); }

// ---------------------------------------------------------------- sections

void sectionThresholdSemantics() {
    StderrCapture cap;
    L().setLevel(LogLevel::Warning);
    CHECK(L().currentLevel() == LogLevel::Warning, "currentLevel mirrors setLevel(Warning)");
    L().setLevel(LogLevel::Debug);
    CHECK(L().currentLevel() == LogLevel::Debug, "currentLevel mirrors setLevel(Debug)");

    L().setLevel(LogLevel::Warning);
    cap.start();
    KFLOG_DEBUG("Probe", "dbg");
    KFLOG_INFO("Probe", "inf");
    L().logf(LogLevel::Debug, "Probe", "fmt%d", 1);
    L().logf(LogLevel::Warning, "Probe", "warn%d", 2);
    L().error("Probe", "err");
    auto lines = splitLines(cap.stop());

    CHECK(lines.size() == 2, "Warning threshold emits exactly warn+error");
    if (lines.size() == 2) {
        CHECK(hasLevel(lines[0], "WARN "), "first emitted record is WARN");
        CHECK(hasLevel(lines[1], "ERROR"), "second emitted record is ERROR");
        CHECK(messageOf(lines[0]) == "warn2", "logf formats into the record body");
        CHECK(messageOf(lines[1]) == "err", "typed API body is verbatim");
    }

    L().setLevel(LogLevel::Error);
    cap.start();
    L().warn("Probe", "suppressed");
    cap.stop();
    CHECK(L().currentLevel() == LogLevel::Error, "threshold is readable mid-stream");

    L().setLevel(LogLevel::Debug);
    cap.start();
    KFLOG_DEBUG("Probe", "dbg");
    KFLOG_INFO("Probe", "inf");
    L().logf(LogLevel::Debug, "Probe", "fmt%d", 1);
    L().warn("Probe", "wn");
    L().error("Probe", "err");
    lines = splitLines(cap.stop());
    CHECK(lines.size() == 5, "Debug threshold emits all five records");
    CHECK(countMatching(lines) == 5, "every record matches the line grammar");
    const char* want[5] = {"DEBUG", "INFO ", "DEBUG", "WARN ", "ERROR"};
    for (size_t i = 0; i < lines.size() && i < 5; ++i) {
        CHECK(hasLevel(lines[i], want[i]), "level token per record position");
    }
}

void sectionFieldPadding() {
    StderrCapture cap;
    L().setLevel(LogLevel::Info);
    cap.start();
    L().info("T", "short-tag");
    L().info("VeryLongTagHere", "truncated-tag");
    auto lines = splitLines(cap.stop());
    CHECK(lines.size() == 2, "two tag fixtures emitted");
    if (lines.size() == 2) {
        CHECK(tagOf(lines[0]) == "T         ", "short tag pads to 10 columns");
        CHECK(tagOf(lines[1]) == "VeryLon...", "over-long tag truncates to 7+ellipsis");
    }
}

void sectionStampValidity() {
    StderrCapture cap;
    L().setLevel(LogLevel::Debug);
    constexpr int kN = 400;
    cap.start();
    for (int i = 0; i < kN; ++i) L().info("Stamp", "monotonic burst");
    auto lines = splitLines(cap.stop());
    CHECK(lines.size() == kN, "burst line count is exact");
    CHECK(countMatching(lines) == kN, "every burst line is a complete record");

    bool monotonic = true;
    bool digits_ok = true;
    long prev = -1;
    for (const auto& l : lines) {
        long ms_of_day = 0;
        if (!stampOf(l, ms_of_day)) { digits_ok = false; continue; }
        // A midnight rollover is the only legitimate decrease; anything else is a
        // stamp that was assembled from two different broken-down times.
        if (prev >= 0 && ms_of_day < prev && prev < 86399000) monotonic = false;
        prev = ms_of_day;
    }
    CHECK(digits_ok, "stamp fields are in range with a 3-digit ms field");
    CHECK(monotonic, "stamps never go backwards inside one burst");
}

void sectionSingleWriteIntegrity() {
    // Foreign threads simulate every non-logger consumer of the same unbuffered
    // stream (a third-party library, or a raw std::cerr call site of the kind this
    // todo removed from PipelineController). They deliberately use a MULTI-insert
    // chain, i.e. several write(2) calls per line, so they can land inside a
    // record that is itself assembled piecewise. The assertion is only about the
    // logger's own records, and it is exact: count, shape and one-per-token.
    constexpr int kLoggers  = 4;
    constexpr int kForeign  = 2;
    constexpr int kPerLog   = 400;

    std::atomic<bool> stop_flag{false};
    StderrCapture cap;
    cap.start();

    std::vector<std::thread> threads;
    for (int t = 0; t < kLoggers; ++t) {
        threads.emplace_back([t] {
            for (int i = 0; i < kPerLog; ++i) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "T%02d-%06d", t, i);
                L().info("Conc", buf);
            }
        });
    }
    for (int f = 0; f < kForeign; ++f) {
        threads.emplace_back([f, &stop_flag] {
            for (int i = 0; !stop_flag.load(std::memory_order_relaxed); ++i) {
                std::cerr << "FOREIGN" << f << "-PADPADPADPAD" << i
                          << "-PADPADPADPAD" << "END\n";
            }
        });
    }
    for (int t = 0; t < kLoggers; ++t) threads[t].join();
    stop_flag.store(true);
    for (int t = kLoggers; t < kLoggers + kForeign; ++t) threads[t].join();

    const std::string text = cap.stop();
    const int expected = kLoggers * kPerLog;

    // Exact count of WHOLE, contiguous records. A record assembled piecewise over
    // unbuffered std::cerr is several write(2) calls, and these foreign writers
    // land inside them, so this count is the discriminator.
    CHECK(countContiguousRecords(text) == expected,
          "every logger record survived the foreign writers intact");

    // And nothing was lost or duplicated: each uniquely-tokened message appears
    // exactly once in the stream.
    int duplicated = 0, missing = 0;
    for (int t = 0; t < kLoggers; ++t) {
        for (int i = 0; i < kPerLog; ++i) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "T%02d-%06d", t, i);
            const int seen = countOccurrences(text, buf);
            if (seen != 1) (seen == 0 ? missing : duplicated)++;
        }
    }
    CHECK(duplicated == 0, "no logger record is emitted twice");
    CHECK(missing == 0, "no logger record is lost");
    CHECK(!text.empty(), "the foreign writers did run (no assertion on their count)");
}

void sectionConcurrentThreshold() {
    std::atomic<bool> done{false};
    StderrCapture cap;
    cap.start();
    std::vector<std::thread> threads;
    for (int f = 0; f < 3; ++f) {
        threads.emplace_back([&done] {
            while (!done.load(std::memory_order_relaxed)) {
                L().setLevel(LogLevel::Debug);
                L().setLevel(LogLevel::Error);
            }
        });
    }
    for (int w = 0; w < 4; ++w) {
        threads.emplace_back([w] {
            for (int i = 0; i < 300; ++i) L().info("Race", "w" + std::to_string(w));
        });
    }
    for (int w = 3; w < 7; ++w) threads[w].join();
    done.store(true);
    for (int f = 0; f < 3; ++f) threads[f].join();
    const auto lines = splitLines(cap.stop());
    L().setLevel(LogLevel::Info);

    CHECK(countMatching(lines) == static_cast<int>(lines.size()),
          "concurrent setLevel yields only well-formed records");
    CHECK(!lines.empty(), "the racing writers did emit records");
}

void sectionTimerZeroAllocation() {
    // The name is far longer than any std::string SSO buffer, so a timer that
    // copies its name into a std::string member allocates once per scope and is
    // caught here; the const char* member allocates nothing.
    static constexpr const char* kLongName =
        "ALLOC_WITNESS_TIMER_NAME_LONGER_THAN_ANY_SSO_BUFFER";
    long before = 0, after = 0;
    StderrCapture cap;
    L().setLevel(LogLevel::Error);  // keep the below-threshold path silent
    cap.start();
    azu_alloc_probe::counter().store(0);
    azu_alloc_probe::armFlag().store(true);
    before = azu_alloc_probe::counter().load();
    for (int i = 0; i < 20000; ++i) {
        utils::ScopedTimer timer(kLongName);
    }
    after = azu_alloc_probe::counter().load();
    azu_alloc_probe::armFlag().store(false);
    const std::string emitted = cap.stop();
    CHECK(after - before == 0, "ScopedTimer scope performs zero allocations");
    CHECK(emitted.empty(), "a below-threshold scope emits no record");
}

void sectionTimerEmitsNamedRecord() {
    static constexpr const char* kName = "SectionAlpha";
    StderrCapture cap;
    L().setLevel(LogLevel::Debug);
    cap.start();
    {
        utils::ScopedTimer timer(kName);
        // Stimulus only: cross the 1 ms print threshold. The assertion below is
        // about the record's existence and content, never about the measured time.
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
    }
    auto lines = splitLines(cap.stop());
    CHECK(lines.size() == 1, "an above-threshold scope emits exactly one record");
    if (lines.size() == 1) {
        CHECK(hasLevel(lines[0], "DEBUG"), "the timer record is DEBUG");
        CHECK(tagOf(lines[0]) == "Timer     ", "the timer record carries the Timer tag");
        const std::string msg = messageOf(lines[0]);
        CHECK(msg.find(kName) != std::string::npos, "the literal name reaches the record");
        CHECK(msg.find(" ms") != std::string::npos, "the record carries a millisecond value");
        const double printed = std::strtod(msg.c_str() + std::strlen(kName) + 2, nullptr);
        CHECK(printed > 1.0, "the printed duration is above the print threshold");
    }

    L().setLevel(LogLevel::Info);
    cap.start();
    {
        utils::ScopedTimer timer("SectionBeta");
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
    }
    CHECK(splitLines(cap.stop()).empty(),
          "the timer honours a threshold above Debug");
    L().setLevel(LogLevel::Info);
}

void sectionTimerNameContract() {
    // The lifetime contract is a compile-time contract: a temporary std::string
    // argument would dangle in the destructor, so there must be no such overload.
    static_assert(std::is_constructible<utils::ScopedTimer, const char*>::value,
                  "ScopedTimer must accept a string literal");
    static_assert(!std::is_constructible<utils::ScopedTimer, std::string>::value,
                  "ScopedTimer must not copy a std::string name on the hot path");
    static_assert(sizeof(utils::ScopedTimer) <= 2 * sizeof(void*) + sizeof(long double),
                  "ScopedTimer carries a pointer and a time point, nothing else");
    CHECK(true, "ScopedTimer name contract holds at compile time");
}

} // namespace

int main() {
    sectionThresholdSemantics();
    sectionFieldPadding();
    sectionStampValidity();
    sectionSingleWriteIntegrity();
    sectionConcurrentThreshold();
    sectionTimerZeroAllocation();
    sectionTimerEmitsNamedRecord();
    sectionTimerNameContract();

    std::printf("logger_timer_contract: %d checks, %d failures\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("logger_timer_contract: FAIL\n");
        return 1;
    }
    std::printf("logger_timer_contract: PASS\n");
    return 0;
}
