#pragma once

#include <vector>
#include <Eigen/Core>
#include <cmath>
#include <mutex>
#include <memory>
#include <string>

namespace kfusion {
namespace meshing {

struct Triangle {
    Eigen::Vector3f v[3];
    Eigen::Vector3f n[3];
    uint8_t         c[3][3]; // RGB per vertex
};

struct MeshData {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> normals;
    std::vector<uint8_t>         colors;    // RGB per vertex (positions.size() * 3)
    std::vector<uint32_t>        indices;   // triangle list

    void clear() {
        positions.clear();
        normals.clear();
        colors.clear();
        indices.clear();
    }

    bool empty() const { return positions.empty(); }
    size_t triangleCount() const { return indices.size() / 3; }

    void reserve(size_t tri_count) {
        positions.reserve(tri_count * 3);
        normals.reserve(tri_count * 3);
        colors.reserve(tri_count * 3 * 3);
        indices.reserve(tri_count * 3);
    }

    void addTriangle(const Triangle& tri) {
        for (int i = 0; i < 3; ++i) {
            uint32_t vidx = static_cast<uint32_t>(positions.size());
            positions.push_back(tri.v[i]);
            normals.push_back(tri.n[i]);
            colors.push_back(tri.c[i][0]);
            colors.push_back(tri.c[i][1]);
            colors.push_back(tri.c[i][2]);
            indices.push_back(vidx);
        }
    }

    // Contract every exporter must satisfy before it touches a file: this mesh can
    // be indexed and serialized without reading out of bounds or emitting
    // non-finite coordinates. Reports the first violated invariant (so the check
    // order below is part of the contract) through `reason`, which may be null.
    // Deterministic, exception-free, and allocation-free on the accepted path.
    // Winding, duplicates, normal normalization and bounds are deliberately not
    // checked: they are not required for a safe write.
    bool validate(std::string* reason) const {
        const size_t nvert = positions.size();
        if (nvert == 0) return reject(reason, "positions is empty");

        const size_t nidx = indices.size();
        if (nidx == 0) {
            return reject(reason, "indices is empty while positions.size()=" + std::to_string(nvert));
        }
        if (nidx % 3 != 0) {
            return reject(reason, "indices.size()=" + std::to_string(nidx) +
                                      " is not a multiple of 3");
        }
        for (size_t i = 0; i < nidx; ++i) {
            const size_t idx = static_cast<size_t>(indices[i]);
            if (idx >= nvert) {
                return reject(reason, "indices[" + std::to_string(i) + "]=" + std::to_string(indices[i]) +
                                      " is out of range for positions.size()=" + std::to_string(nvert));
            }
        }
        for (size_t i = 0; i < nvert; ++i) {
            if (!componentsAreFinite(positions[i])) {
                return reject(reason, "positions[" + std::to_string(i) + "]=" + summarize(positions[i]) +
                                      " has a non-finite component");
            }
        }

        const size_t nnorm = normals.size();
        if (nnorm != 0 && nnorm != nvert) {
            return reject(reason, "normals.size()=" + std::to_string(nnorm) +
                                      " is neither 0 nor positions.size()=" + std::to_string(nvert));
        }
        for (size_t i = 0; i < nnorm; ++i) {
            if (!componentsAreFinite(normals[i])) {
                return reject(reason, "normals[" + std::to_string(i) + "]=" + summarize(normals[i]) +
                                      " has a non-finite component");
            }
        }

        // Division, not multiplication: colors.size() == nvert * 3 is tested as
        // (colors.size() % 3 == 0 && colors.size() / 3 == nvert), so no size_t
        // product can overflow for any input.
        const size_t ncol = colors.size();
        if (ncol != 0 && (ncol % 3 != 0 || ncol / 3 != nvert)) {
            return reject(reason, "colors.size()=" + std::to_string(ncol) +
                                      " is neither 0 nor 3 * positions.size() (" +
                                      std::to_string(nvert) + " vertices)");
        }
        return true;
    }

private:
    static bool reject(std::string* reason, const std::string& why) {
        if (reason != nullptr) *reason = why;
        return false;
    }

    static bool componentsAreFinite(const Eigen::Vector3f& v) {
        return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
    }

    static std::string summarize(const Eigen::Vector3f& v) {
        return "(" + std::to_string(v.x()) + ", " + std::to_string(v.y()) + ", " +
               std::to_string(v.z()) + ")";
    }
};

// Thread-safe mesh container used between extraction thread and render thread
class SharedMesh {
public:
    void update(std::shared_ptr<MeshData> mesh) {
        std::lock_guard<std::mutex> lk(mutex_);
        mesh_ = std::move(mesh);
        version_++;
    }

    // Returns shared pointer for zero-copy rendering access
    std::shared_ptr<MeshData> snapshot(uint64_t& version_out) const {
        std::lock_guard<std::mutex> lk(mutex_);
        version_out = version_;
        return mesh_;
    }

    uint64_t version() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return version_;
    }

private:
    mutable std::mutex        mutex_;
    std::shared_ptr<MeshData> mesh_;
    uint64_t           version_ = 0;
};

} // namespace meshing
} // namespace kfusion
