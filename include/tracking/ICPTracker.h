#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>
#include <vector>
#include "sensor/FrameData.h"
#ifdef CUDA_ENABLED
#include "utils/CudaUniquePtr.h"
#endif
#ifdef HIP_ENABLED
#include "utils/HipUniquePtr.h"
#include <hip/hip_vector_types.h>
#endif

namespace kfusion {
namespace tracking {

struct ICPParams {
    int    max_iterations[sensor::FramePyramid::LEVELS] = {10, 5, 4};
    float  dist_threshold    = 0.1f;  // meters
    float  angle_threshold   = 30.0f; // degrees
    float  min_depth         = 0.3f;
    float  max_depth         = 5.0f;
};

struct ICPResult {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    float           error      = 0.0f;
    int             inliers    = 0;
    bool            converged  = false;
    bool            tracking_ok = false;
    float           final_step = std::numeric_limits<float>::infinity();

    // Undamped point-to-plane information matrix (J^T W J) of the last linear
    // system built at the finest level solved, in the increment coordinates
    // [t; omega] of the live camera. Its small eigen-directions are the motions
    // the scene cannot observe (e.g. sliding along a single wall).
    Eigen::Matrix<float, 6, 6> information = Eigen::Matrix<float, 6, 6>::Zero();

    // Diagnostic counters
    int             valid_live_points  = 0;
    int             valid_model_points = 0;
    int             projected_points   = 0;
    int             dist_filtered      = 0;
    int             angle_filtered     = 0;
};

// Raycasted model points and normals used as reference for ICP
struct ModelFrame {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;
    std::vector<uint8_t>         colors;
    
#ifdef CUDA_ENABLED
    utils::CudaUniquePtr<float3> d_vertices;
    utils::CudaUniquePtr<float3> d_normals;
    utils::CudaUniquePtr<uchar3> d_colors;
#elif defined(HIP_ENABLED)
    utils::HipUniquePtr<float3> d_vertices;
    utils::HipUniquePtr<float3> d_normals;
    utils::HipUniquePtr<uchar3> d_colors;
#endif

    int width  = sensor::FRAME_W;
    int height = sensor::FRAME_H;

    // World-from-camera pose this model image was raycast at. ICP must use it as
    // the reference pose: the model lags the live camera by at least one
    // integrated frame, and projecting live points with any other pose looks
    // them up at the wrong model pixels.
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    uint64_t        source_frame_id = 0;

    ModelFrame() {
        vertices.assign(width * height, Eigen::Vector3f::Zero());
        normals.assign(width * height, Eigen::Vector3f::Zero());
        colors.assign(width * height * 3, 0);
    }
    
    // Use default destructor as CudaUniquePtr handles cleanup
};

class ICPTracker {
public:
    explicit ICPTracker(const ICPParams& params = ICPParams{});

    ICPResult track(const sensor::FramePyramid& live,
                    const ModelFrame&           model,
                    const Eigen::Matrix4f&      pose_estimate,
                    const Eigen::Matrix4f&      ref_pose);

#ifdef CUDA_ENABLED
    ICPResult trackGPU(const float*                d_depth,
                       const uint8_t*              d_rgb,
                       int                         width,
                       int                         height,
                       const ModelFrame&           model,
                       const Eigen::Matrix4f&      pose_estimate,
                       const Eigen::Matrix4f&      ref_pose);
    void initGPU();
    void freeGPU();
#elif defined(HIP_ENABLED)
    ICPResult trackGPU(const float*                d_depth,
                       const uint8_t*              d_rgb,
                       int                         width,
                       int                         height,
                       const ModelFrame&           model,
                       const Eigen::Matrix4f&      pose_estimate,
                       const Eigen::Matrix4f&      ref_pose);
    void initGPU();
    void freeGPU();
#endif

    const ICPParams& params() const { return params_; }
    void setParams(const ICPParams& p) { params_ = sanitizeParams(p); }
    void setNumThreads(int n) { num_threads_.store(n); }

private:
    static ICPParams sanitizeParams(const ICPParams& params);

    ICPParams params_;
    std::atomic<int> num_threads_{0};

    ICPResult trackLevel(const sensor::FrameData& live_level,
                         const ModelFrame&        model,
                         const Eigen::Matrix4f&   pose_estimate,
                         const Eigen::Matrix4f&   ref_pose,
                         int                      level,
                         int                      max_iter);

#ifdef CUDA_ENABLED
    // Persistent GPU buffers to avoid allocations
    utils::CudaUniquePtr<float> d_hessian_; 
    
    // Pyramid level buffers
    utils::CudaUniquePtr<float3> d_pyramid_v[sensor::FramePyramid::LEVELS];
    utils::CudaUniquePtr<float3> d_pyramid_n[sensor::FramePyramid::LEVELS];
    
    ICPResult trackLevelGPU(const float3*            d_v_live,
                            const float3*            d_n_live,
                            int                      width,
                            int                      height,
                            const ModelFrame&        model,
                            const Eigen::Matrix4f&   pose_estimate,
                            const Eigen::Matrix4f&   ref_pose,
                            int                      level,
                            int                      max_iter);
#elif defined(HIP_ENABLED)
    // Persistent GPU buffers to avoid allocations
    utils::HipUniquePtr<float> d_hessian_; 
    
    // Pyramid level buffers
    utils::HipUniquePtr<float3> d_pyramid_v[sensor::FramePyramid::LEVELS];
    utils::HipUniquePtr<float3> d_pyramid_n[sensor::FramePyramid::LEVELS];
    
    ICPResult trackLevelGPU(const float3*            d_v_live,
                            const float3*            d_n_live,
                            int                      width,
                            int                      height,
                            const ModelFrame&        model,
                            const Eigen::Matrix4f&   pose_estimate,
                            const Eigen::Matrix4f&   ref_pose,
                            int                      level,
                            int                      max_iter);
#endif

    // Point-to-plane linear system build
    bool buildLinearSystem(const sensor::FrameData& live,
                           const ModelFrame&        model,
                           const Eigen::Matrix4f&   pose,
                           const Eigen::Matrix4f&   ref_pose,
                           Eigen::Matrix<float,6,6>& A,
                           Eigen::Matrix<float,6,1>& b,
                           float& residual,
                           int&   inlier_count,
                           int&   valid_live,
                           int&   valid_model,
                            int&   projected,
                            int&   dist_filtered,
                            int&   angle_filtered);
    // tracking:CPU-7: there is deliberately no separate model-projection member
    // here — projection lives inline in `buildLinearSystem`, at
    // full-resolution intrinsics on every path.
};

} // namespace tracking
} // namespace kfusion
