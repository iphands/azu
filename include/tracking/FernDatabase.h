#pragma once

// Random-fern keyframe database for relocalization (relocalization rework,
// step 6). Glocker, Shotton, Criminisi, Izadi, "Real-Time RGB-D Camera
// Relocalization via Randomized Ferns for Keyframe Encoding", TVCG 2015, as
// used by InfiniTAM v3 (Kähler et al., ECCV 2016) and ElasticFusion.
//
// A frame is encoded as a compact code:
//   - depth is reduced to a 40x30 thumbnail (block means over valid pixels) and
//     blurred with a Gaussian (sigma 2.5 thumbnail px), as in Kähler et al.;
//   - each of 500 ferns applies 4 fixed random tests "thumbnail[pixel] < threshold"
//     (pixel and threshold drawn once from a seeded generator; invalid depth
//     counts as far, i.e. the test is false), giving a 4-bit code per fern.
// Two frames are compared by BlockHD, the share of ferns whose codes differ: 0 for
// the same view, ~1 for unrelated views. Views that look alike get alike codes,
// so while tracking is good the pipeline stores a keyframe (code + pose) whenever
// the frame is unlike every stored one (BlockHD > 0.2), and while lost it looks
// up the nearest keyframes and tries their poses. The lookup is a vote over
// inverted code tables (500 table rows per query), not a scan of every keyframe.
// Depth only: RGB is not registered to depth in this pipeline.
//
// Retrieval only proposes poses; relocalization verifies them
// (tracking/Relocalizer.h). Tracking thread only; not thread-safe.

#include <Eigen/Core>

#include <cstdint>
#include <utility>
#include <vector>

namespace kfusion {
namespace tracking {

struct FernParams {
    int      ferns           = 500;
    int      tests_per_fern  = 4;       // <= 8: one byte per fern code
    int      thumb_w         = 40;
    int      thumb_h         = 30;
    float    sigma           = 2.5f;    // thumbnail px
    float    min_valid_share = 0.25f;   // of a block's pixels for the block to count
    float    min_depth       = 0.3f;    // test thresholds are drawn from this band (m)
    float    max_depth       = 4.0f;
    uint32_t seed            = 0x5eed1ceu;
    float    add_threshold   = 0.2f;    // add a keyframe when BlockHD to all > this
    int      capacity        = 1000;
};

using FernCode = std::vector<uint8_t>;   // one code per fern

class FernEncoder {
public:
    explicit FernEncoder(const FernParams& p = FernParams{});

    // Depth in meters (0 = none), w x h. The thumbnail takes w/thumb_w x
    // h/thumb_h block means when the size divides evenly, else nearest samples.
    FernCode encode(const float* depth, int w, int h) const;
    // The blurred thumbnail the tests read (0 = no depth nearby), thumb_w x thumb_h.
    std::vector<float> thumbnail(const float* depth, int w, int h) const;

    const FernParams& params() const { return p_; }

private:
    struct Test {
        uint16_t pixel;
        float    threshold;
    };
    FernParams        p_;
    std::vector<Test> tests_;    // ferns * tests_per_fern
};

struct Keyframe {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    uint64_t        frame_id = 0;
    FernCode        code;
};

class FernDatabase {
public:
    explicit FernDatabase(const FernParams& p = FernParams{});

    const FernEncoder& encoder() const { return enc_; }
    const FernParams&  params() const { return enc_.params(); }
    size_t size() const { return kfs_.size(); }
    const Keyframe& keyframe(size_t i) const { return kfs_[i]; }
    void clear();

    // Share of ferns whose codes differ: 0 identical, 1 all different.
    static float blockHD(const FernCode& a, const FernCode& b);
    // BlockHD of `q` to every keyframe, by voting over the code tables.
    std::vector<float> dissimilarities(const FernCode& q) const;
    // Up to k keyframes, most similar first: (index, BlockHD).
    std::vector<std::pair<int, float>> nearest(const FernCode& q, int k) const;
    // 1 when the database is empty.
    float minDissimilarity(const FernCode& q) const;
    // Store `q` as a keyframe when it is unlike every stored one (BlockHD >
    // add_threshold) and there is room. Returns whether it was added.
    bool maybeAdd(const FernCode& q, const Eigen::Matrix4f& pose, uint64_t frame_id);

private:
    FernEncoder                   enc_;
    std::vector<Keyframe>         kfs_;
    std::vector<std::vector<int>> table_;   // [fern * 2^tests + code] -> keyframe indices
};

} // namespace tracking
} // namespace kfusion
