// make_fake_dump: writes a libfakenect recording generated from the synthetic
// room (tests/support/SyntheticScene.h), for smoke_fakenect.
//
// Timestamps are what a real Kinect v1 produces: 60 MHz ticks, depth every
// 2,002,155 ticks and RGB every 2,000,287 (so the two streams slip phase), RGB
// offset by 19.8 ms, starting 2 s before the uint32 counter wraps. The replay
// therefore exercises the exact timestamp handling c311c69 broke.
//
// usage: make_fake_dump <out_dir> [frames]
#include "sensor/DepthValidity.h"
#include "support/SyntheticScene.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: make_fake_dump <out_dir> [frames]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const int frames = argc > 2 ? std::atoi(argv[2]) : 60;
    mkdir(dir.c_str(), 0755);

    constexpr uint64_t kDepthPeriod = 2002155, kRgbPeriod = 2000287;
    constexpr uint64_t kStart = (1ull << 32) - 120000000ull;   // 2 s before the wrap
    constexpr uint64_t kRgbOffset = 1188000;                    // 19.8 ms
    constexpr double kTicksPerSecond = 60e6;

    const azu_test::SyntheticScene scene;
    FILE* index = std::fopen((dir + "/INDEX.txt").c_str(), "w");
    if (!index) return 1;

    int di = 0, ri = 0;
    while (di < frames) {
        const uint64_t td = kStart + static_cast<uint64_t>(di) * kDepthPeriod;
        const uint64_t tr = kStart + kRgbOffset + static_cast<uint64_t>(ri) * kRgbPeriod;
        const bool depth = td <= tr;
        const uint64_t t = depth ? td : tr;
        char name[96];
        std::snprintf(name, sizeof(name), "%c-%.6f-%u-%s", depth ? 'd' : 'r',
                      static_cast<double>(t) / kTicksPerSecond, static_cast<uint32_t>(t),
                      depth ? "0.pgm" : "0.ppm");
        FILE* f = std::fopen((dir + "/" + name).c_str(), "wb");
        if (!f) return 1;
        if (depth) {
            // A slow sideways pan out and back, so libfakenect's loop replay
            // continues smoothly instead of teleporting the camera home.
            const float ph = static_cast<float>(di < frames / 2 ? di : frames - di);
            const Eigen::Matrix4f pose =
                azu_test::makePose({0.0f, 0.003f * ph, 0.0f}, {0.0015f * ph, 0.0f, 0.0f});
            const std::vector<float> m = scene.renderDepth(pose);
            std::vector<uint16_t> raw(m.size());
            for (size_t i = 0; i < m.size(); ++i)
                raw[i] = m[i] > 0.0f ? kfusion::sensor::cpuDepthMetersToRaw(m[i], 0.3f, 5.0f) : 0;
            std::fprintf(f, "P5 640 480 65535\n");
            std::fwrite(raw.data(), sizeof(uint16_t), raw.size(), f);   // little-endian host order
            ++di;
        } else {
            std::vector<uint8_t> rgb(640 * 480 * 3);
            for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<uint8_t>((i * 7 + ri) & 0xff);
            std::fprintf(f, "P6 640 480 255\n");
            std::fwrite(rgb.data(), 1, rgb.size(), f);
            ++ri;
        }
        std::fclose(f);
        std::fprintf(index, "%s\n", name);
    }
    std::fclose(index);
    std::printf("make_fake_dump: %d depth + %d rgb frames in %s\n", di, ri, dir.c_str());
    return 0;
}
