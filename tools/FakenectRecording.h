// Reading recordings written by libfreenect's fakenect-record (fakenect/record.c).
//
// A recording directory holds INDEX.txt, one file name per line in capture
// order, and the files it names:
//   d-<host s>-<ticks>-<...>.pgm   11-bit depth: "P5 640 480 65535\n" + 640x480 uint16 LE
//   r-<host s>-<ticks>-<...>.ppm   RGB: "P6 640 480 255\n" + 640x480x3 bytes
//   a-<host s>-<ticks>-<...>.dump  raw freenect_raw_tilt_state; the first three
//                                   int16 are accelerometer counts (819 = 1 g)
//   device.json                     the unit's factory registration
// <host s> is the recording host's clock in seconds; <ticks> is the device's
// 60 MHz timestamp (a copies the last frame's).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace azu_rec {

inline constexpr int    kWidth       = 640;
inline constexpr int    kHeight      = 480;
inline constexpr double kCountsPerG  = 819.0;   // FREENECT_COUNTS_PER_G

struct Entry {
    char        type = 0;      // 'd' depth, 'r' rgb, 'a' accelerometer
    double      host_s = 0.0;
    uint32_t    ticks = 0;
    std::string file;          // name relative to the recording directory
};

// Entries of <dir>/INDEX.txt in file order; lines that do not parse are skipped.
// `ok` (optional) is false when INDEX.txt cannot be opened.
inline std::vector<Entry> parseIndex(const std::string& dir, bool* ok = nullptr) {
    std::vector<Entry> out;
    std::ifstream index(dir + "/INDEX.txt");
    if (ok) *ok = static_cast<bool>(index);
    std::string line;
    while (std::getline(index, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        Entry e;
        unsigned int ticks = 0;
        if (line.size() < 3 || std::sscanf(line.c_str(), "%c-%lf-%u-", &e.type, &e.host_s, &ticks) != 3) {
            continue;
        }
        e.ticks = ticks;
        e.file = line;
        out.push_back(std::move(e));
    }
    return out;
}

// Frame payload after the one-line PGM/PPM header.
inline bool readPayload(const std::string& path, size_t bytes, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string header;
    std::getline(f, header);
    out.resize(bytes);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(f.gcount()) == bytes;
}

inline bool readDepth(const std::string& dir, const Entry& e, std::vector<uint16_t>& out) {
    std::vector<uint8_t> bytes;
    if (!readPayload(dir + "/" + e.file, static_cast<size_t>(kWidth) * kHeight * 2, bytes)) return false;
    out.resize(static_cast<size_t>(kWidth) * kHeight);
    std::copy(bytes.begin(), bytes.end(), reinterpret_cast<uint8_t*>(out.data()));   // little-endian host
    return true;
}

inline bool readAccelCounts(const std::string& dir, const Entry& e, int16_t xyz[3]) {
    std::ifstream f(dir + "/" + e.file, std::ios::binary);
    return static_cast<bool>(f.read(reinterpret_cast<char*>(xyz), 3 * sizeof(int16_t)));
}

}  // namespace azu_rec
