// marching_cubes_table_contract (big-fix todo 6): an independent, first-principles
// crossing-edge oracle for all 256 rows of the shared CPU edge table
// (kfusion::meshing::tables::edge_table in include/meshing/MarchingCubesTables.h).
//
// FAIL-BEFORE-FIX: this test is EXPECTED to be RED at todo 6 and GREEN only after
// todo 7 repairs the three known-corrupt rows. The current CPU/shared table carries
//   edge_table[213] = 0x835 (canonical 0x83f)
//   edge_table[214] = 0xb3f (canonical 0xb35)
//   edge_table[215] = 0xa36 (canonical 0xa3c)
// This oracle recomputes every mask from corner sign patterns alone, so it names
// exactly those three rows and nothing else. It never reads uninitialized memory,
// never depends on a sanitizer, timing, or undefined behavior: it is a pure,
// deterministic integer comparison over a frozen constant table.
//
// ORACLE, from first principles (NOT parity, NOT symmetry, NOT self-comparison):
// A Marching Cubes cube has 8 corners (bit c of the cube configuration is set when
// corner c is on the "inside" / negative side of the isosurface) and 12 edges. An
// edge is a CROSSING edge exactly when its two endpoint corners lie on opposite
// sides, i.e. when the sign bits of its two corners differ. The edge_table entry for
// a configuration is therefore the 12-bit mask whose bit e is set iff edge e crosses.
// The endpoint-corner pairs below are the very ones the production extraction path
// consumes (EDGE_CORNERS in src/meshing/MarchingCubes.cpp), so the oracle validates
// the table against the corner indexing the meshing stage actually uses.
//
// The crossing predicate is sign-symmetric (it only tests whether the two endpoint
// bits differ), so the oracle is invariant to which literal sign the codebase calls
// "inside": it validates the geometry of each mask, not a sign convention.

#include "meshing/MarchingCubesTables.h"

#include <cstdio>

namespace {

int g_failures = 0;

// Endpoint corner index of each of the 12 cube edges. Mirrors EDGE_CORNERS in
// src/meshing/MarchingCubes.cpp so the oracle and the consumer agree on edge
// numbering; this is the only structural fact borrowed from the consumer, and the
// correctness verdict comes entirely from the independent crossing predicate.
constexpr int kEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

// Rebuild the crossing-edge mask for one configuration from the corner sign bits.
constexpr int crossingEdgeMask(int config) {
    int mask = 0;
    for (int e = 0; e < 12; ++e) {
        const int a = (config >> kEdgeCorners[e][0]) & 1;
        const int b = (config >> kEdgeCorners[e][1]) & 1;
        if (a != b) mask |= (1 << e);
    }
    return mask;
}

// Every mask bit must lie inside the 12-edge field.
constexpr int kEdgeFieldMask = (1 << 12) - 1;

void testEdgeTableAgainstCrossingOracle() {
    using kfusion::meshing::tables::edge_table;

    int mismatch_count = 0;
    for (int config = 0; config < 256; ++config) {
        const int expected = crossingEdgeMask(config);
        const int actual   = edge_table[config];

        if ((actual & ~kEdgeFieldMask) != 0) {
            std::printf("FAIL: edge_table[%d] = 0x%x sets bits outside the 12-edge field\n",
                        config, actual);
            ++g_failures;
            ++mismatch_count;
        }
        if (actual != expected) {
            std::printf("FAIL: edge_table[%d] = 0x%x, crossing-edge oracle expects 0x%x\n",
                        config, actual, expected);
            ++g_failures;
            ++mismatch_count;
        }
    }

    if (mismatch_count != 0) {
        std::printf("crossing-edge oracle: %d row(s) disagree with the shared CPU edge table\n",
                    mismatch_count);
    }
}

// A handful of hand-verified anchors keep the oracle honest against a silent bug in
// the oracle itself (each is derivable by counting the crossing edges of the pattern).
void testOracleKnownAnchors() {
    // config 0x00: no corner inside -> no crossing edge.
    if (crossingEdgeMask(0x00) != 0x000) { std::printf("FAIL: oracle anchor 0x00\n"); ++g_failures; }
    // config 0x01: only corner 0 inside -> its three edges {0,3,8} cross.
    if (crossingEdgeMask(0x01) != 0x109) { std::printf("FAIL: oracle anchor 0x01\n"); ++g_failures; }
    // config 0xFF: all corners inside -> no crossing edge.
    if (crossingEdgeMask(0xFF) != 0x000) { std::printf("FAIL: oracle anchor 0xFF\n"); ++g_failures; }
    // The canonical values for the three corrupt rows, asserted independently so the
    // red verdict is attributable to the table, not to a mis-derived oracle.
    if (crossingEdgeMask(213) != 0x83f) { std::printf("FAIL: oracle anchor 213\n"); ++g_failures; }
    if (crossingEdgeMask(214) != 0xb35) { std::printf("FAIL: oracle anchor 214\n"); ++g_failures; }
    if (crossingEdgeMask(215) != 0xa3c) { std::printf("FAIL: oracle anchor 215\n"); ++g_failures; }
}

} // namespace

int main() {
    testOracleKnownAnchors();
    testEdgeTableAgainstCrossingOracle();
    if (g_failures == 0) {
        std::printf("marching_cubes_table_contract: PASS (256 rows vs independent crossing-edge oracle)\n");
        return 0;
    }
    std::printf("marching_cubes_table_contract: FAIL (%d failed check(s); expected RED before big-fix todo 7)\n",
                g_failures);
    return 1;
}
