// CPU PLY writer (big-fix todo 30). Binary and ASCII share ONE schema and ONE
// record-writing path (PLY-07): the element/property lines are identical except
// the single "format" token, and the vertex/face records are emitted by the same
// template through a small Sink, so the two formats can no longer drift. The mesh
// is fully validated BEFORE any file is opened, so a rejected mesh creates no file
// (PLY-05 never yields a partial color stream; PLY-04 is never silently truncated).
// The binary payload is declared and emitted explicitly little-endian (PLY-02/03);
// a big-endian host is refused rather than written wrong-endian. Every opened file
// is flushed, explicitly closed and checked, and any post-open fault removes the
// partial file (PLY-01/08). Records are built into one contiguous buffer and
// written in a single stream call instead of per-field churn (PLY-06).
//
// Reconciliation with MeshData::validate(): a raw index count that is not a
// multiple of 3, and a color buffer that is neither empty nor exactly one RGB
// triple per vertex, are mesh-contract violations. They are REPORTED and REFUSED
// (no file, no silently-dropped remainder, no partial color stream) rather than
// silently truncated -- the conservative reading of "explicitly discarded, never
// hidden". A file is only ever produced when the counts already agree, so its
// declared vertex/face counts and per-record layout always match its bytes.

#include "export/PLYExporter.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace kfusion {
namespace export_io {
namespace {

// The host byte order, resolved once. The binary path is the only consumer.
bool hostIsLittleEndian() {
    union { uint32_t u; unsigned char b[4]; } probe;
    probe.u = 1u;
    return probe.b[0] == 1u;
}

// Explicit little-endian serialization: byte-order independent by construction,
// so the bytes on disk match the "binary_little_endian" header on any host.
void appendU32LE(std::string& out, uint32_t v) {
    out.push_back(static_cast<char>(v & 0xFFu));
    out.push_back(static_cast<char>((v >> 8) & 0xFFu));
    out.push_back(static_cast<char>((v >> 16) & 0xFFu));
    out.push_back(static_cast<char>((v >> 24) & 0xFFu));
}
void appendFloatLE(std::string& out, float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    appendU32LE(out, u);
}

// Shortest text that round-trips a float (%.9g over the widened double).
std::string fmtFloat(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return std::string(buf);
}

struct PlyPlan {
    size_t nvert       = 0;
    size_t nface       = 0;   // exactly indices.size()/3, only reached when divisible
    bool   has_normals = false;
    bool   has_colors  = false;
};

// Validate through the mesh contract. On failure, report the cause (with an
// explicit note for the two classes this todo owns) and create nothing.
bool buildPlan(const meshing::MeshData& mesh, const std::string& filepath, PlyPlan& plan) {
    std::string reason;
    if (mesh.validate(&reason)) {
        plan.nvert       = mesh.positions.size();
        plan.nface       = mesh.indices.size() / 3;   // multiple of 3 is guaranteed here
        plan.has_normals = (mesh.normals.size() == plan.nvert);
        plan.has_colors  = (mesh.colors.size() == plan.nvert * 3);
        return true;
    }
    std::cerr << "[PLY] Rejected " << filepath << ": " << reason << "\n";
    if (reason.find("multiple of 3") != std::string::npos) {
        std::cerr << "[PLY]   trailing indices are only whole triangles short; refusing "
                     "to silently truncate the mesh.\n";
    } else if (reason.find("colors.size()") != std::string::npos) {
        std::cerr << "[PLY]   a partial color stream is never written.\n";
    }
    return false;
}

// One shared header: identical element/property lines for both formats, the only
// difference being the format token itself.
std::string headerText(const PlyPlan& p, const char* format) {
    std::string h = "ply\n";
    h += "format "; h += format; h += " 1.0\n";
    h += "comment KinectFusionQt export\n";
    h += "element vertex " + std::to_string(p.nvert) + "\n";
    h += "property float x\nproperty float y\nproperty float z\n";
    if (p.has_normals) h += "property float nx\nproperty float ny\nproperty float nz\n";
    if (p.has_colors)  h += "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    h += "element face " + std::to_string(p.nface) + "\n";
    h += "property list uchar int vertex_indices\n";
    h += "end_header\n";
    return h;
}

// Binary Sink: appends little-endian bytes straight into the payload buffer.
struct BinarySink {
    std::string& out;
    explicit BinarySink(std::string& o) : out(o) {}
    void beginRecord() {}
    void endRecord() {}
    void floatField(float v)   { appendFloatLE(out, v); }
    void byteField(uint8_t v)  { out.push_back(static_cast<char>(v)); }
    void indexField(uint32_t v){ appendU32LE(out, v); }
};

// ASCII Sink: whitespace-separated fields, one record per line.
struct AsciiSink {
    std::string& out;
    bool         first = true;
    explicit AsciiSink(std::string& o) : out(o) {}
    void sep() { if (!first) out.push_back(' '); first = false; }
    void beginRecord() { first = true; }
    void endRecord()   { out.push_back('\n'); }
    void floatField(float v)   { sep(); out += fmtFloat(v); }
    void byteField(uint8_t v)  { sep(); out += std::to_string(static_cast<int>(v)); }
    void indexField(uint32_t v){ sep(); out += std::to_string(v); }
};

// The single record-writing path both formats share. The header counts and this
// iteration walk the same plan, so declared counts and emitted records agree.
template <typename Sink>
void writeRecords(Sink& sink, const meshing::MeshData& mesh, const PlyPlan& plan) {
    for (size_t i = 0; i < plan.nvert; ++i) {
        sink.beginRecord();
        sink.floatField(mesh.positions[i].x());
        sink.floatField(mesh.positions[i].y());
        sink.floatField(mesh.positions[i].z());
        if (plan.has_normals) {
            sink.floatField(mesh.normals[i].x());
            sink.floatField(mesh.normals[i].y());
            sink.floatField(mesh.normals[i].z());
        }
        if (plan.has_colors) {
            sink.byteField(mesh.colors[i * 3 + 0]);
            sink.byteField(mesh.colors[i * 3 + 1]);
            sink.byteField(mesh.colors[i * 3 + 2]);
        }
        sink.endRecord();
    }
    for (size_t t = 0; t < plan.nface; ++t) {
        sink.beginRecord();
        sink.byteField(static_cast<uint8_t>(3));
        sink.indexField(mesh.indices[t * 3 + 0]);
        sink.indexField(mesh.indices[t * 3 + 1]);
        sink.indexField(mesh.indices[t * 3 + 2]);
        sink.endRecord();
    }
}

// Best-effort cleanup of a file we opened but failed to finish. Never throws out
// of the writer; a failed removal is simply the OS's own state.
void removePartial(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Shared write tail: build the whole output in memory, open, write once, flush,
// close, and on any fault after creation remove the file and report failure.
bool writeFile(const std::string& out, const std::string& filepath, bool binary) {
    std::ofstream f(filepath, binary ? (std::ios::binary | std::ios::trunc)
                                      : std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "[PLY] Cannot open: " << filepath << "\n";
        return false;   // nothing was created by us
    }
    f.write(out.data(), static_cast<std::streamsize>(out.size()));
    f.flush();
    if (!f.good()) {
        std::cerr << "[PLY] Write error on: " << filepath << "\n";
        f.close();
        removePartial(filepath);
        return false;
    }
    f.close();
    if (f.fail()) {
        std::cerr << "[PLY] Close error on: " << filepath << "\n";
        removePartial(filepath);
        return false;
    }
    return true;
}

} // namespace

bool PLYExporter::writeBinary(const meshing::MeshData& mesh, const std::string& filepath) {
    if (!hostIsLittleEndian()) {
        std::cerr << "[PLY] Big-endian host: refusing to emit a little-endian payload to "
                  << filepath << " (would write wrong-endian bytes).\n";
        return false;   // before opening: no file is created
    }
    PlyPlan plan;
    if (!buildPlan(mesh, filepath, plan)) return false;   // before opening: no file

    std::string body;
    BinarySink sink(body);
    writeRecords(sink, mesh, plan);                       // contiguous, single buffer
    const std::string out = headerText(plan, "binary_little_endian") + body;

    if (!writeFile(out, filepath, /*binary=*/true)) return false;
    std::cout << "[PLY] Exported " << plan.nvert << " vertices, " << plan.nface
              << " triangles to: " << filepath << "\n";
    return true;
}

bool PLYExporter::writeASCII(const meshing::MeshData& mesh, const std::string& filepath) {
    PlyPlan plan;
    if (!buildPlan(mesh, filepath, plan)) return false;   // before opening: no file

    std::string body;
    AsciiSink sink(body);
    writeRecords(sink, mesh, plan);                       // same record path as binary
    const std::string out = headerText(plan, "ascii") + body;

    if (!writeFile(out, filepath, /*binary=*/false)) return false;
    std::cout << "[PLY] ASCII exported " << plan.nvert << " vertices, " << plan.nface
              << " triangles to: " << filepath << "\n";
    return true;
}

} // namespace export_io
} // namespace kfusion
