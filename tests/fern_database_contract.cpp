// fern_database_contract (relocalization rework, step 6): the random-fern
// keyframe database in include/tracking/FernDatabase.h, on depth rendered from
// the synthetic room (camera at the room centre, as in pipeline_spin_contract).
//
//   A  the tests are fixed by the seed: same seed, same codes; another seed differs
//   B  BlockHD: identical codes 0, all-different codes 1; no depth gives code 0
//   C  the code-table vote equals brute-force BlockHD on random codes
//   D  insertion: the first frame is added, the same view is not, a view 60 deg
//      away is; the capacity holds
//   E  retrieval: with keyframes every 15 deg of a full turn, a view 5 deg past a
//      keyframe finds that keyframe in the top 3 always, and first in >= 75% of
//      cases. The synthetic room is nearly bare walls, so codes separate weakly
//      (views 60 deg apart differ in only ~30% of ferns); measured top-1 19/24.
//      The pipeline tries the top 3 and verifies each, so top 3 is what counts.
#include "support/SyntheticScene.h"
#include "tracking/FernDatabase.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

namespace {

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
    } while (false)

using namespace kfusion::tracking;
constexpr float kDeg = 3.14159265f / 180.0f;

class Views {
public:
    std::vector<float> at(float yaw_deg, float pitch_deg = -5.0f) const {
        const Eigen::Matrix4f w = azu_test::makePose({0.0f, yaw_deg * kDeg, 0.0f}, {0, 0, 0}) *
                                  azu_test::makePose({pitch_deg * kDeg, 0.0f, 0.0f}, {0, 0, 0});
        return scene_.renderDepth(room_ * w, K_);
    }
    const azu_test::Intrinsics& K() const { return K_; }

private:
    azu_test::SyntheticScene scene_;
    azu_test::Intrinsics K_;
    Eigen::Matrix4f room_ = azu_test::makePose({0, 0, 0}, {0.0f, -0.05f, 1.15f});
};

}  // namespace

int main() {
    const Views views;
    const auto& K = views.K();
    FernParams params;
    params.max_depth = 3.0f;

    // A
    const std::vector<float> v0 = views.at(0.0f);
    const FernEncoder e1(params), e2(params);
    FernParams other = params;
    other.seed = params.seed + 1;
    const FernEncoder e3(other);
    const FernCode c1 = e1.encode(v0.data(), K.width, K.height);
    CHECK(c1 == e2.encode(v0.data(), K.width, K.height), "A: same seed, same code");
    CHECK(FernDatabase::blockHD(c1, e3.encode(v0.data(), K.width, K.height)) > 0.3f,
          "A: another seed, another code");
    CHECK(c1.size() == 500, "A: 500 ferns");

    // B
    CHECK(FernDatabase::blockHD(c1, c1) == 0.0f, "B: identical codes 0");
    FernCode flipped = c1;
    for (auto& c : flipped) c = static_cast<uint8_t>(c ^ 0x0f);
    CHECK(FernDatabase::blockHD(c1, flipped) == 1.0f, "B: all-different codes 1");
    const std::vector<float> none(v0.size(), 0.0f);
    const FernCode empty = e1.encode(none.data(), K.width, K.height);
    bool all_zero = true;
    for (uint8_t c : empty) all_zero = all_zero && c == 0;
    CHECK(all_zero, "B: no depth -> every test false");

    // C
    {
        FernParams tp = params;
        tp.add_threshold = -1.0f;   // store everything
        FernDatabase db(tp);
        std::mt19937 rng(7);
        std::vector<FernCode> codes;
        for (int i = 0; i < 40; ++i) {
            FernCode c(500);
            for (auto& x : c) x = static_cast<uint8_t>(rng() & 0x0f);
            codes.push_back(c);
            db.maybeAdd(c, Eigen::Matrix4f::Identity(), static_cast<uint64_t>(i));
        }
        FernCode q(500);
        for (auto& x : q) x = static_cast<uint8_t>(rng() & 0x0f);
        const std::vector<float> d = db.dissimilarities(q);
        bool same = d.size() == codes.size();
        for (size_t i = 0; same && i < d.size(); ++i)
            same = std::fabs(d[i] - FernDatabase::blockHD(q, codes[i])) < 1e-6f;
        CHECK(same, "C: code-table vote == brute-force BlockHD");
    }

    // D
    {
        FernParams dp = params;
        dp.capacity = 3;
        FernDatabase db(dp);
        CHECK(db.maybeAdd(c1, Eigen::Matrix4f::Identity(), 1), "D: first frame added");
        CHECK(!db.maybeAdd(e1.encode(views.at(1.0f).data(), K.width, K.height), Eigen::Matrix4f::Identity(), 2),
              "D: the same view (1 deg later) is not added");
        const FernCode far = e1.encode(views.at(60.0f).data(), K.width, K.height);
        std::printf("  D: BlockHD 0 vs 1 deg %.3f, 0 vs 60 deg %.3f\n",
                    FernDatabase::blockHD(c1, e1.encode(views.at(1.0f).data(), K.width, K.height)),
                    FernDatabase::blockHD(c1, far));
        CHECK(db.maybeAdd(far, Eigen::Matrix4f::Identity(), 3), "D: a view 60 deg away is added");
        CHECK(db.maybeAdd(e1.encode(views.at(150.0f).data(), K.width, K.height), Eigen::Matrix4f::Identity(), 4) &&
                  !db.maybeAdd(e1.encode(views.at(240.0f).data(), K.width, K.height), Eigen::Matrix4f::Identity(), 5),
              "D: the capacity (3) holds");
        CHECK(db.size() == 3, "D: three keyframes");
    }

    // E
    {
        FernParams ep = params;
        ep.add_threshold = -1.0f;
        FernDatabase db(ep);
        for (int i = 0; i < 24; ++i) {
            const float yaw = 15.0f * static_cast<float>(i);
            db.maybeAdd(e1.encode(views.at(yaw).data(), K.width, K.height), Eigen::Matrix4f::Identity(),
                        static_cast<uint64_t>(i));
        }
        int top1 = 0, top3 = 0;
        for (int i = 0; i < 24; ++i) {
            const float yaw = 15.0f * static_cast<float>(i) + 5.0f;
            const auto hits = db.nearest(e1.encode(views.at(yaw).data(), K.width, K.height), 3);
            if (!hits.empty() && hits[0].first == i) ++top1;
            for (const auto& h : hits) top3 += h.first == i;
        }
        std::printf("  E: 24 queries 5 deg past a keyframe: top-1 %d, top-3 %d\n", top1, top3);
        CHECK(top1 >= 18, "E: the nearest keyframe first in >= 75%");
        CHECK(top3 == 24, "E: the nearest keyframe in the top 3 always");
    }

    if (g_failures == 0) {
        std::printf("fern_database_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("fern_database_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
