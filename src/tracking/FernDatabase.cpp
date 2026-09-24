#include "tracking/FernDatabase.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace kfusion {
namespace tracking {

FernEncoder::FernEncoder(const FernParams& p) : p_(p) {
    // Raw mt19937 output, not std::uniform_*_distribution: those are not
    // specified identically across standard libraries, and the tests (and any
    // stored codes) depend on the exact draw.
    std::mt19937 rng(p_.seed);
    const uint32_t pixels = static_cast<uint32_t>(p_.thumb_w * p_.thumb_h);
    tests_.resize(static_cast<size_t>(p_.ferns) * p_.tests_per_fern);
    for (Test& t : tests_) {
        t.pixel = static_cast<uint16_t>(rng() % pixels);
        const double u = static_cast<double>(rng()) / 4294967296.0;
        t.threshold = p_.min_depth + static_cast<float>(u) * (p_.max_depth - p_.min_depth);
    }
}

std::vector<float> FernEncoder::thumbnail(const float* depth, int w, int h) const {
    const int tw = p_.thumb_w, th = p_.thumb_h;
    std::vector<float> small(static_cast<size_t>(tw) * th, 0.0f);
    std::vector<float> valid(small.size(), 0.0f);
    const bool blocks = w % tw == 0 && h % th == 0;
    const int bw = blocks ? w / tw : 1, bh = blocks ? h / th : 1;
    const int min_count = std::max(1, static_cast<int>(std::ceil(p_.min_valid_share * bw * bh)));
    for (int ty = 0; ty < th; ++ty) {
        for (int tx = 0; tx < tw; ++tx) {
            float sum = 0.0f;
            int n = 0;
            if (blocks) {
                for (int y = ty * bh; y < (ty + 1) * bh; ++y)
                    for (int x = tx * bw; x < (tx + 1) * bw; ++x) {
                        const float z = depth[static_cast<size_t>(y) * w + x];
                        if (z > 0.0f && std::isfinite(z)) { sum += z; ++n; }
                    }
            } else {
                const int x = std::min(w - 1, static_cast<int>((tx + 0.5f) * w / tw));
                const int y = std::min(h - 1, static_cast<int>((ty + 0.5f) * h / th));
                const float z = depth[static_cast<size_t>(y) * w + x];
                if (z > 0.0f && std::isfinite(z)) { sum = z; n = 1; }
            }
            if (n >= min_count) {
                small[static_cast<size_t>(ty) * tw + tx] = sum / static_cast<float>(n);
                valid[static_cast<size_t>(ty) * tw + tx] = 1.0f;
            }
        }
    }
    // Separable Gaussian as a normalized convolution over valid pixels only, so
    // holes neither pull depth toward 0 nor spread as zeros.
    const int r = static_cast<int>(std::ceil(3.0f * p_.sigma));
    std::vector<float> k(2 * r + 1);
    for (int i = -r; i <= r; ++i) k[i + r] = std::exp(-0.5f * i * i / (p_.sigma * p_.sigma));
    auto blur = [&](const std::vector<float>& in, bool horizontal) {
        std::vector<float> out(in.size(), 0.0f);
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
                float acc = 0.0f;
                for (int i = -r; i <= r; ++i) {
                    const int xx = horizontal ? x + i : x, yy = horizontal ? y : y + i;
                    if (xx < 0 || yy < 0 || xx >= tw || yy >= th) continue;
                    acc += k[i + r] * in[static_cast<size_t>(yy) * tw + xx];
                }
                out[static_cast<size_t>(y) * tw + x] = acc;
            }
        return out;
    };
    std::vector<float> weighted(small.size());
    for (size_t i = 0; i < small.size(); ++i) weighted[i] = small[i] * valid[i];
    const std::vector<float> num = blur(blur(weighted, true), false);
    const std::vector<float> den = blur(blur(valid, true), false);
    std::vector<float> out(small.size(), 0.0f);
    for (size_t i = 0; i < out.size(); ++i) out[i] = den[i] > 1e-6f ? num[i] / den[i] : 0.0f;
    return out;
}

FernCode FernEncoder::encode(const float* depth, int w, int h) const {
    const std::vector<float> t = thumbnail(depth, w, h);
    FernCode code(static_cast<size_t>(p_.ferns), 0);
    for (int f = 0; f < p_.ferns; ++f) {
        uint8_t c = 0;
        for (int b = 0; b < p_.tests_per_fern; ++b) {
            const Test& test = tests_[static_cast<size_t>(f) * p_.tests_per_fern + b];
            const float z = t[test.pixel];
            // No depth counts as far: the test is false.
            if (z > 0.0f && z < test.threshold) c = static_cast<uint8_t>(c | (1u << b));
        }
        code[f] = c;
    }
    return code;
}

FernDatabase::FernDatabase(const FernParams& p)
    : enc_(p), table_(static_cast<size_t>(p.ferns) << p.tests_per_fern) {}

void FernDatabase::clear() {
    kfs_.clear();
    for (auto& row : table_) row.clear();
}

float FernDatabase::blockHD(const FernCode& a, const FernCode& b) {
    if (a.empty() || a.size() != b.size()) return 1.0f;
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
    return static_cast<float>(diff) / static_cast<float>(a.size());
}

std::vector<float> FernDatabase::dissimilarities(const FernCode& q) const {
    const int ferns = enc_.params().ferns;
    const int shift = enc_.params().tests_per_fern;
    std::vector<int> equal(kfs_.size(), 0);
    if (static_cast<int>(q.size()) == ferns) {
        for (int f = 0; f < ferns; ++f) {
            for (int id : table_[(static_cast<size_t>(f) << shift) + q[f]]) ++equal[id];
        }
    }
    std::vector<float> d(kfs_.size());
    for (size_t i = 0; i < d.size(); ++i) {
        d[i] = 1.0f - static_cast<float>(equal[i]) / static_cast<float>(ferns);
    }
    return d;
}

std::vector<std::pair<int, float>> FernDatabase::nearest(const FernCode& q, int k) const {
    const std::vector<float> d = dissimilarities(q);
    std::vector<std::pair<int, float>> all;
    all.reserve(d.size());
    for (size_t i = 0; i < d.size(); ++i) all.emplace_back(static_cast<int>(i), d[i]);
    const size_t n = std::min(all.size(), static_cast<size_t>(std::max(0, k)));
    std::partial_sort(all.begin(), all.begin() + static_cast<long>(n), all.end(),
                      [](const auto& a, const auto& b) {
                          return a.second < b.second || (a.second == b.second && a.first < b.first);
                      });
    all.resize(n);
    return all;
}

float FernDatabase::minDissimilarity(const FernCode& q) const {
    const std::vector<float> d = dissimilarities(q);
    return d.empty() ? 1.0f : *std::min_element(d.begin(), d.end());
}

bool FernDatabase::maybeAdd(const FernCode& q, const Eigen::Matrix4f& pose, uint64_t frame_id) {
    const FernParams& p = enc_.params();
    if (static_cast<int>(q.size()) != p.ferns) return false;
    if (static_cast<int>(kfs_.size()) >= p.capacity) return false;
    if (!(minDissimilarity(q) > p.add_threshold)) return false;
    const int id = static_cast<int>(kfs_.size());
    kfs_.push_back(Keyframe{pose, frame_id, q});
    for (int f = 0; f < p.ferns; ++f) {
        table_[(static_cast<size_t>(f) << p.tests_per_fern) + q[f]].push_back(id);
    }
    return true;
}

} // namespace tracking
} // namespace kfusion
