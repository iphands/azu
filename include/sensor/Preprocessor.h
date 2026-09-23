#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "sensor/KinectSensor.h"
#include "sensor/SignalConditioner.h"

namespace kfusion {
namespace sensor {

enum class PreprocessBackend {
    Auto,
    CPU,
    CUDA,
    HIP
};

const char* backendName(PreprocessBackend backend);
PreprocessBackend parseBackendName(const std::string& name);

class Preprocessor {
public:
    virtual ~Preprocessor() = default;

    virtual void reset() = 0;
    virtual void resetTemporalState() = 0;
    virtual void process(RawFrame& frame, float min_depth_m, float max_depth_m) = 0;
    virtual void setSrScale(int scale) = 0;
    // Opt-in upscale (off by default; only the CPU backend implements one).
    virtual void setUpscaleEnabled(bool) {}

    // Upscaled-RGB availability contract (big-fix Todo 22), mirroring
    // SignalConditioner::srUpscaledAvailable(). getSrRgbUpscaled() is valid only
    // while this is true, so the CPU-only upscaled path fails closed here too:
    // the defaults below are the contract for every non-CPU backend, which
    // implements no upscaled pass and must report unavailable rather than be
    // compiled or run to find out.
    virtual bool srUpscaledAvailable() const { return false; }
    virtual uint64_t srUpscaledFrameId() const { return 0; }
    virtual bool srUpscaledAvailableForFrame(uint64_t frame_id) const {
        return srUpscaledAvailable() && srUpscaledFrameId() == frame_id;
    }

    virtual const std::vector<uint8_t>& getSrRgbUpscaled() const = 0;

    // GPU-native access (returns nullptr if CPU backend is used)
    virtual float*    getGPUDepthMeters() const { return nullptr; }
    virtual uint8_t*  getGPURgb()         const { return nullptr; }

    virtual PreprocessBackend backend() const = 0;
};

class CPUPreprocessor final : public Preprocessor {
public:
    CPUPreprocessor();

    void reset() override;
    void resetTemporalState() override;
    void process(RawFrame& frame, float min_depth_m, float max_depth_m) override;
    void setSrScale(int scale) override { conditioner_.setSrScale(scale); }
    void setUpscaleEnabled(bool enabled) override { conditioner_.setUpscaleEnabled(enabled); }
    bool srUpscaledAvailable() const override { return conditioner_.srUpscaledAvailable(); }
    uint64_t srUpscaledFrameId() const override { return conditioner_.srUpscaledFrameId(); }
    bool srUpscaledAvailableForFrame(uint64_t frame_id) const override {
        return conditioner_.srUpscaledAvailableForFrame(frame_id);
    }
    const std::vector<uint8_t>& getSrRgbUpscaled() const override { return conditioner_.getSrRgbUpscaled(); }
    PreprocessBackend backend() const override { return PreprocessBackend::CPU; }

private:
    SignalConditioner conditioner_;
};

class CUDAPreprocessor final : public Preprocessor {
public:
    explicit CUDAPreprocessor(cudaStream_t stream);

    void reset() override;
    void resetTemporalState() override;
    void process(RawFrame& frame, float min_depth_m, float max_depth_m) override;
    void setSrScale(int scale) override { conditioner_.setSrScale(scale); }
    // Deliberately NOT overridden: the CUDA/HIP conditioner implements no EASU
    // upscaled pass (see docs/CUDA_HIP_DEFERRED_CHANGES.md), so this backend keeps
    // the base fail-closed defaults and reports the upscaled buffer unavailable
    // even when its own getSrRgbUpscaled() still hands out preallocated bytes.
    const std::vector<uint8_t>& getSrRgbUpscaled() const override { return conditioner_.getSrRgbUpscaled(); }
    PreprocessBackend backend() const override { return PreprocessBackend::CUDA; }

    float*   getGPUDepthMeters() const override { 
#ifdef CUDA_ENABLED
        return conditioner_.getGPUDepthMeters(); 
#elif defined(HIP_ENABLED)
        return conditioner_.getGPUDepthMeters(); 
#else
        return nullptr;
#endif
    }
    uint8_t* getGPURgb()         const override { 
#ifdef CUDA_ENABLED
        return conditioner_.getGPURgb(); 
#elif defined(HIP_ENABLED)
        return conditioner_.getGPURgb(); 
#else
        return nullptr;
#endif
    }

private:
    cudaStream_t stream_ = nullptr;
    SignalConditioner conditioner_;
};

std::unique_ptr<Preprocessor> makePreprocessor(PreprocessBackend requested_backend,
                                               bool cuda_available,
                                               cudaStream_t cuda_stream,
                                               PreprocessBackend* actual_backend = nullptr);

} // namespace sensor
} // namespace kfusion
