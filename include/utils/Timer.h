#pragma once
#include <chrono>

namespace kfusion {
namespace utils {

// RAII scope timer. `name` is stored as a pointer, NOT copied into a
// std::string: a ScopedTimer is constructed on the hot path (per-frame TSDF
// integration, per-frame mesh extraction), where a std::string copy is a heap
// allocation whose only purpose is to be read once in the destructor. The name
// must therefore outlive the timer: pass a string literal or another static
// string object. A temporary std::string argument would dangle and is a
// compile error, because there is deliberately no std::string overload.
class ScopedTimer {
public:
    explicit ScopedTimer(const char* name)
        : name_(name ? name : "?"), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer();

private:
    const char* name_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace utils
} // namespace kfusion
