#include "utils/Timer.h"
#include "utils/Logger.h"

namespace kfusion {
namespace utils {

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    // Only print if > 1ms to reduce noise, using the logger for consistency.
    // Nothing below runs (and nothing allocates) until a scope actually crosses
    // that threshold, which is why the name is a plain const char*.
    if (ms > 1.0) {
        KFLOGF_DEBUG("Timer", "%s: %.2f ms", name_, ms);
    }
}

} // namespace utils
} // namespace kfusion
