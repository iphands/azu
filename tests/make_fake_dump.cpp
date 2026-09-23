// make_fake_dump: writes a libfakenect recording generated from the synthetic
// room (tests/support/SyntheticScene.h), for smoke_fakenect.
//
// Timestamps are what a real Kinect v1 produces: 60 MHz ticks, depth every
// 2,002,155 ticks and RGB every 2,000,287 (so the two streams slip phase), RGB
// offset by 19.8 ms, starting 2 s before the uint32 counter wraps. The replay
// therefore exercises the exact timestamp handling c311c69 broke.
//
// Modes: "pan" (default; a slow out-and-back pan, loop-replay friendly) and
// "spin" (camera at the room centre turning a full 360 degrees at 1.5 deg per
// frame, then holding still -- the capture that broke on real hardware). Both
// also write accelerometer records ('a' files, gravity only) like the real
// recorder, so replay tools can check tilt.
//
// usage: make_fake_dump <out_dir> [frames] [pan|spin]
#include "sensor/DepthValidity.h"
#include "support/SyntheticScene.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
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
    const bool spin = argc > 3 && std::string(argv[3]) == "spin";
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
            Eigen::Matrix4f pose;
            if (spin) {
                constexpr int kTurn = 240;   // 240 x 1.5 deg = 360 deg
                const float yaw = 1.5f * 3.14159265f / 180.0f * static_cast<float>(std::min(di, kTurn));
                pose = azu_test::makePose({0.0f, 0.0f, 0.0f}, {0.0f, -0.05f, 1.15f}) *
                       azu_test::makePose({0.0f, yaw, 0.0f}, {0.0f, 0.0f, 0.0f});
            } else {
                // A slow sideways pan out and back, so libfakenect's loop replay
                // continues smoothly instead of teleporting the camera home.
                const float ph = static_cast<float>(di < frames / 2 ? di : frames - di);
                pose = azu_test::makePose({0.0f, 0.003f * ph, 0.0f}, {0.0015f * ph, 0.0f, 0.0f});
            }
            const std::vector<float> m = scene.renderDepth(pose);
            std::vector<uint16_t> raw(m.size());
            for (size_t i = 0; i < m.size(); ++i)
                raw[i] = m[i] > 0.0f ? kfusion::sensor::cpuDepthMetersToRaw(m[i], 0.3f, 5.0f) : 0;
            std::fprintf(f, "P5 640 480 65535\n");
            std::fwrite(raw.data(), sizeof(uint16_t), raw.size(), f);   // little-endian host order
            ++di;
            // Accelerometer record after the frame, like fakenect-record: a
            // freenect_raw_tilt_state (3 x int16 counts, int8 tilt, pad, int32
            // status). Level camera, yaw only: gravity stays on +y at 819
            // counts/g, as the real unit reports when level.
            std::fclose(f);
            char aname[96];
            std::snprintf(aname, sizeof(aname), "a-%.6f-%u-0.dump",
                          static_cast<double>(t) / kTicksPerSecond + 0.001, static_cast<uint32_t>(t));
            FILE* af = std::fopen((dir + "/" + aname).c_str(), "wb");
            if (!af) return 1;
            unsigned char tilt[12] = {0};
            const int16_t accel[3] = {0, 819, 0};
            std::memcpy(tilt, accel, sizeof(accel));
            std::fwrite(tilt, 1, sizeof(tilt), af);
            std::fclose(af);
            std::fprintf(index, "%s\n", name);
            std::fprintf(index, "%s\n", aname);
            continue;
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
