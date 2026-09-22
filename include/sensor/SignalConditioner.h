#pragma once

#include <cstdint>
#include <vector>

#ifdef CUDA_ENABLED
#include "utils/CudaUniquePtr.h"
#include <cuda_runtime_api.h>
#elif defined(HIP_ENABLED)
#include "utils/HipUniquePtr.h"
#include <hip/hip_runtime_api.h>
#endif

namespace kfusion {
namespace sensor {

struct RawFrame;

#ifdef CUDA_ENABLED
using cudaStream_t = ::cudaStream_t;
#elif defined(HIP_ENABLED)
using cudaStream_t = ::hipStream_t;
#else
using cudaStream_t = void*;
#endif

class SignalConditioner {
public:
    SignalConditioner();

    void reset();
    void resetEMA();
    void process(RawFrame& raw, cudaStream_t cuda_stream, float min_depth_m, float max_depth_m);
    void setSrScale(int scale) { sr_scale_ = scale; }
    int getSrScale() const { return sr_scale_; }

    // ---- upscaled-RGB availability contract (big-fix Todo 22) ----
    // The ONLY gate on sr_rgb_upscaled_. The raw getter below is not a validity
    // signal: the constructor preallocates the buffer with FRAME_W * FRAME_H * 3
    // zero bytes, so a consumer that reads it unconditionally textures TSDF with
    // black — the "black textures" complaint that made the only product consumer
    // (src/app/PipelineController.cpp, trackingLoop) dead in the first place.
    // Every consumer must check srUpscaledAvailable() first and treat the bytes
    // as undefined otherwise.
    //
    // The contract is CPU-only and fail-closed:
    //   - false on a freshly constructed object and after reset()
    //   - false before the first successful process()
    //   - false whenever the frame geometry or the scale is invalid
    //     (only 2 <= getSrScale() <= 4 can ever be an upscaled buffer; scale 1
    //     is an original-resolution image, not an upscale, and is never published)
    //   - false for the whole of the current frame as soon as process() starts,
    //     so a size-mismatch early return or a failed CPU stage can never leave
    //     the PREVIOUS frame's bytes readable as if they were fresh
    //   - false on the GPU path: processCuda() implements no upscaled pass and
    //     never publishes, so a GPU backend reports the buffer unavailable
    //     without any GPU code being compiled or invoked to find that out
    //     (docs/CUDA_HIP_DEFERRED_CHANGES.md keeps the backend upscaled hazards
    //     deferred)
    // True only after the CPU upscale stage has fully produced a fresh buffer of
    // exactly FRAME_W * sr_scale * FRAME_H * sr_scale * 3 bytes for `raw`.
    // Definitions live in SignalConditioner_omp.cpp because the geometry check
    // needs FRAME_W / FRAME_H.
    bool srUpscaledAvailable() const;
    // The raw.frame_id the currently published buffer was produced from. Only
    // meaningful while srUpscaledAvailable() is true; reset() clears it to 0.
    uint64_t srUpscaledFrameId() const { return sr_upscaled_frame_id_; }
    // Availability pinned to one frame id, for consumers that hold the frame they
    // processed and must not texture from a later one.
    bool srUpscaledAvailableForFrame(uint64_t frame_id) const;

    // Valid ONLY while srUpscaledAvailable() is true; see the contract above.
    const std::vector<uint8_t>& getSrRgbUpscaled() const { return sr_rgb_upscaled_; }

private:
    int sr_scale_ = 2; // Default 2x upscaling
    // Availability state for sr_rgb_upscaled_. Never consulted on its own:
    // srUpscaledAvailable() re-derives the geometry and scale every call, so a
    // stale flag can never outlive the buffer it describes.
    bool sr_upscaled_available_ = false;
    uint64_t sr_upscaled_frame_id_ = 0;
    std::vector<float>    ema_buf_m_;
    std::vector<uint8_t>  sr_rgb_; // For guidance (original resolution)
    std::vector<uint8_t>  sr_rgb_upscaled_; // For TSDF texturing (upscaled)
    std::vector<uint8_t>  rgb_scratch_;
    std::vector<float>    guidance_luma_;
    std::vector<uint16_t> depth_scratch_;
    // Temporal-depth EMA destination/source split (big-fix Todo 20). The other
    // depth passes already ping-pong through depth_scratch_; the EMA used to
    // write its result back into the very array its 8-neighbour reset test
    // reads, so a thread's answer depended on which neighbours had already been
    // updated. depth_src_ is the immutable input snapshot of the current frame
    // and ema_buf_m_ is the per-pixel (index-private) temporal state; the
    // caller's buffer is the destination. Never read a neighbour from the
    // destination.
    std::vector<uint16_t> depth_src_;

    void preprocessRgb(std::vector<uint8_t>& rgb);
    void buildSuperResolutionGuidance(const std::vector<uint8_t>& rgb);
    // Publishes sr_rgb_upscaled_ through the availability contract; `frame_id` is
    // raw.frame_id, recorded only for a buffer that was actually produced.
    void applySuperResolutionToRgb(const std::vector<uint8_t>& rgb, uint64_t frame_id);
    void invalidateUpscaled();
    void denoiseDepthSpatial(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m);
    void applyDepthEma(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m);
    void fillDepthHoles(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m);
    void guidedDepthFilter(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m);
    void processCpu(RawFrame& raw, float min_depth_m, float max_depth_m);

#ifdef CUDA_ENABLED
    bool processCuda(RawFrame& raw, cudaStream_t cuda_stream, float min_depth_m, float max_depth_m);

    // GPU resources
    utils::CudaUniquePtr<uint8_t>  d_rgb_in_;
    utils::CudaUniquePtr<uint8_t>  d_rgb_out_;
    utils::CudaUniquePtr<uint16_t> d_depth_in_;
    utils::CudaUniquePtr<uint16_t> d_depth_out_;
    utils::CudaUniquePtr<float>    d_depth_meters_;
    utils::CudaUniquePtr<float>    d_ema_buf_m_;
    utils::CudaUniquePtr<float>    d_guidance_luma_;

public:
    uint16_t* getGPUDepthRaw() const { return d_depth_out_.get(); }
    float*    getGPUDepthMeters() const { return d_depth_meters_.get(); }
    uint8_t*  getGPURgb() const { return d_rgb_out_.get(); }
#elif defined(HIP_ENABLED)
    bool processCuda(RawFrame& raw, cudaStream_t cuda_stream, float min_depth_m, float max_depth_m);

    // GPU resources
    utils::HipUniquePtr<uint8_t>  d_rgb_in_;
    utils::HipUniquePtr<uint8_t>  d_rgb_out_;
    utils::HipUniquePtr<uint16_t> d_depth_in_;
    utils::HipUniquePtr<uint16_t> d_depth_out_;
    utils::HipUniquePtr<float>    d_depth_meters_;
    utils::HipUniquePtr<float>    d_ema_buf_m_;
    utils::HipUniquePtr<float>    d_guidance_luma_;

public:
    uint16_t* getGPUDepthRaw() const { return d_depth_out_.get(); }
    float*    getGPUDepthMeters() const { return d_depth_meters_.get(); }
    uint8_t*  getGPURgb() const { return d_rgb_out_.get(); }
#endif

public:
#ifdef AZU_PIPELINE_TEST_SEAM
    // ---- Headless CPU depth-stage test seam (compile-time, test targets only) ----
    // Same mechanism and same discipline as the PipelineController seam: the
    // macro is defined only for the azu_test_core/azu_test_pipeline targets, so
    // production never sees these. They add member FUNCTIONS only (no layout
    // change) and drive the real depth stages with no device, display, GPU or
    // stream. Without the seam the CPU depth contracts could only reach the
    // whole processCpu() chain, where the median/guided-filter/RGB passes
    // dominate the fixture and the hole-fill / EMA invariants stop being
    // observable.
    void fillDepthHolesForTests(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
        fillDepthHoles(depth, min_depth_m, max_depth_m);
    }
    void applyDepthEmaForTests(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
        applyDepthEma(depth, min_depth_m, max_depth_m);
    }
    // Read-only view of the per-pixel temporal state (0.0f == no history).
    const std::vector<float>& emaStateForTests() const { return ema_buf_m_; }

    // ---- CPU CAS guidance test seam (big-fix Todo 21) ----
    // Same discipline as the depth seam above: member FUNCTIONS only, no layout
    // change, production never sees them. They drive the REAL
    // buildSuperResolutionGuidance() (RCAS sharpening of sr_rgb_ and the luma
    // reduction) with an injected RGB buffer, so the contract tests pin the
    // product's CAS + guidance-luma arithmetic instead of a reimplementation of
    // it. `rgb` must be exactly FRAME_W * FRAME_H * 3 bytes, which is what
    // processCpu() hands the real stage.
    void buildGuidanceForTests(const std::vector<uint8_t>& rgb) {
        buildSuperResolutionGuidance(rgb);
    }
    // Post-CAS guidance image (original resolution) and the luma derived from it.
    const std::vector<uint8_t>& guidanceRgbForTests() const { return sr_rgb_; }
    const std::vector<float>& guidanceLumaForTests() const { return guidance_luma_; }
#endif
};

} // namespace sensor
} // namespace kfusion
