#pragma once

// Canonical CPU depth-domain validity boundary (big-fix Todo 20).
//
// ONE place owns the raw-depth predicate, the reciprocal curve, the configured
// meter band and the rejected-output value. Every CPU surface that turns a raw
// 11-bit Kinect code into meters, or meters back into a raw code, consumes this
// header instead of open-coding a range check, because the predicate alone is
// NOT a sufficient validity gate:
//
//   meters(raw) = 1 / (raw * -0.0030711016 + 3.3309495161)
//
// has a reciprocal pole at raw = 3.3309495161 / 0.0030711016 ~= 1084.61, which
// sits INSIDE the 11-bit band. So a raw value that passes the predicate can
// still convert to +533 m (raw 1084) or to a NEGATIVE distance (raw 1085 ->
// -836.3 m, raw 2046 -> -0.3387 m). Raw validity and band validity are two
// different questions and only the conjunction of the two is usable depth.
//
// Canonical rule (docs/CANONICAL_SEMANTICS.md, section "Depth domain"):
//
//   raw invalid:      raw == 0 || raw >= 2047
//   raw valid:        raw != 0 && raw < 2047
//   accepted meters:  finite(meters) && meters >= min_depth && meters <= max_depth
//   rejected output:  exactly 0.0f (meters) / exactly raw 0 (codes)
//
// Rejection NEVER clamps. An out-of-band measurement is dropped to the invalid
// sentinel; it is not snapped to min_depth, to max_depth, to raw 1 or to raw
// 2046. Snapping is what made an unreachable distance look like a real wall,
// and raw 2046 decodes to a negative distance, so a high-side clamp could even
// manufacture geometry behind the camera plane.
//
// Header-only, device-free, allocation-free and side-effect-free: it is the
// seam the CPU contracts in tests/depth_domain_contract.cpp and
// tests/depth_ema_determinism_contract.cpp assert against. The CUDA/HIP
// preprocessing kernels have their own device copies and are NOT touched here;
// their parity is documentary and deferred (docs/CUDA_HIP_DEFERRED_CHANGES.md,
// rows cross-backend:A5, sensor:S-01/S-02, sensor:S-04, sensor:S-19).

#include <cmath>
#include <cstdint>

namespace kfusion {
namespace sensor {

// First raw code the sensor uses for "no valid depth returned". Codes at or
// above it are invalid, so the predicate is `>=` and not `==`: the sensor is
// 11-bit, but a wider or corrupted buffer must not smuggle 2048+ through.
constexpr uint16_t kRawDepthSentinel = 2047;

// Raw codes a valid measurement may actually occupy.
constexpr uint16_t kRawDepthValidMin = 1;
constexpr uint16_t kRawDepthValidMax = 2046;

// Kinect v1 11-bit -> meters reciprocal curve: 1 / (raw * kRawDepthCurveA + kRawDepthCurveB).
constexpr float kRawDepthCurveA = -0.0030711016f;
constexpr float kRawDepthCurveB = 3.3309495161f;

// Canonical raw predicate. raw 0 (no data) and raw >= 2047 (sensor sentinel)
// are invalid; nothing else in this header decides raw validity.
inline bool isInvalidRawDepth(uint16_t raw) {
    return raw == 0 || raw >= kRawDepthSentinel;
}

inline bool isValidRawDepth(uint16_t raw) {
    return !isInvalidRawDepth(raw);
}

// Raw -> meters for the accepted raw codes. Invalid raw becomes exactly 0.0f.
// The returned value is NOT yet usable depth: it still has to clear the
// configured band, because of the pole at raw ~= 1084.61. Callers that are
// about to hand depth to geometry/tracking/integration must use
// cpuDepthMeters() below instead of calling this and gating by hand.
inline float rawDepthToMeters(uint16_t raw) {
    if (isInvalidRawDepth(raw)) return 0.0f;
    return 1.0f / (static_cast<float>(raw) * kRawDepthCurveA + kRawDepthCurveB);
}

// Configured-band predicate over the meter domain. Non-finite is never in
// band: NaN fails both comparisons already, +/-Inf is caught explicitly so the
// rule reads as the three-way conjunction the docs state.
inline bool isDepthMetersInBand(float meters, float min_depth_m, float max_depth_m) {
    if (!std::isfinite(meters)) return false;
    return meters >= min_depth_m && meters <= max_depth_m;
}

// THE CPU depth boundary. Raw code in, downstream-ready meters out: exactly
// 0.0f whenever the raw code is invalid OR its converted meters fall outside
// the configured band. No clamping, ever.
inline float cpuDepthMeters(uint16_t raw, float min_depth_m, float max_depth_m) {
    if (isInvalidRawDepth(raw)) return 0.0f;
    const float meters = rawDepthToMeters(raw);
    return isDepthMetersInBand(meters, min_depth_m, max_depth_m) ? meters : 0.0f;
}

// Meters -> raw code, inverse of the same curve, with the SAME rejection rule
// on the code side: a meter value that is non-finite or out of band yields raw
// 0 (invalid) rather than the nearest in-range code. The `raw` round-trip is
// additionally required to land on an actually valid raw code, so this function
// can never emit 0-by-clamp for a high-side overflow and can never emit 2047.
inline uint16_t cpuDepthMetersToRaw(float meters, float min_depth_m, float max_depth_m) {
    if (!isDepthMetersInBand(meters, min_depth_m, max_depth_m)) return 0;
    const float raw = (1.0f / meters - kRawDepthCurveB) / kRawDepthCurveA;
    if (!std::isfinite(raw)) return 0;
    const long code = std::lround(raw);
    if (code < static_cast<long>(kRawDepthValidMin) || code > static_cast<long>(kRawDepthValidMax)) {
        return 0;
    }
    return static_cast<uint16_t>(code);
}

// True when a raw code carries a measurement that is usable right now: valid
// raw AND in-band meters. Hole fill and every filtering window use this as the
// neighbour predicate, which is what keeps raw 2047 (a sentinel whose meters
// convert to 0.0f) out of the interpolatable set.
inline bool isUsableRawDepth(uint16_t raw, float min_depth_m, float max_depth_m) {
    return !isInvalidRawDepth(raw) &&
           isDepthMetersInBand(rawDepthToMeters(raw), min_depth_m, max_depth_m);
}

} // namespace sensor
} // namespace kfusion
