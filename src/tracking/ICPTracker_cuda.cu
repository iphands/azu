#ifdef CUDA_ENABLED

#include "tracking/ICPTracker.h"
#include "tracking/ICPShared.h"
#include "sensor/KinectSensor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>  // SelfAdjointEigenSolver (:423) + LDLT (:434); matches HIP twin

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

namespace kfusion {
namespace tracking {



// ---------------------------------------------------------------------------
// GPU-Based Vertex and Normal Generation
// ---------------------------------------------------------------------------
__global__ void computeVerticesKernel(const float* depth, float3* vertices, int w, int h, float fx, float fy, float cx, float cy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    float d = depth[idx];
    if (d > 0.0f) vertices[idx] = make_float3((x - cx) / fx * d, (y - cy) / fy * d, d);
    else vertices[idx] = make_float3(0, 0, 0);
}

__global__ void computeNormalsKernel(const float* depth, const float3* vertices, float3* normals, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) {
        if (x < w && y < h) normals[y * w + x] = make_float3(0, 0, 0);
        return;
    }
    int c = y * w + x;
    float dc = depth[c];
    if (dc <= 0.0f || depth[c+1] <= 0.0f || depth[c-1] <= 0.0f || depth[c-w] <= 0.0f || depth[c+w] <= 0.0f) {
        normals[c] = make_float3(0, 0, 0); return;
    }
    float jump = fmaxf(0.03f, dc * 0.05f);
    if (fabsf(dc-depth[c+1]) > jump || fabsf(dc-depth[c-1]) > jump || fabsf(dc-depth[c-w]) > jump || fabsf(dc-depth[c+w]) > jump) {
        normals[c] = make_float3(0, 0, 0); return;
    }
    float3 dx = make_float3(vertices[c+1].x - vertices[c-1].x, vertices[c+1].y - vertices[c-1].y, vertices[c+1].z - vertices[c-1].z);
    float3 dy = make_float3(vertices[c+w].x - vertices[c-w].x, vertices[c+w].y - vertices[c-w].y, vertices[c+w].z - vertices[c-w].z);
    float3 n = make_float3(dx.y*dy.z - dx.z*dy.y, dx.z*dy.x - dx.x*dy.z, dx.x*dy.y - dx.y*dy.x);
    float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    normals[c] = (len > 1e-6f) ? make_float3(n.x/len, n.y/len, n.z/len) : make_float3(0, 0, 0);
}

// ---------------------------------------------------------------------------
// GPU-Based Pyramid Generation (Downsampling)
// ---------------------------------------------------------------------------
__global__ void downsampleKernel(
    const float3* v_src, const float3* n_src,
    int src_w, int src_h,
    float3* v_dst, float3* n_dst,
    int dst_w, int dst_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    int sx = x * 2;
    int sy = y * 2;

    float3 v_sum = make_float3(0,0,0);
    float3 n_sum = make_float3(0,0,0);
    int count = 0;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int sidx = (sy + dy) * src_w + (sx + dx);
            float3 v = v_src[sidx];
            float3 n = n_src[sidx];
            if (v.z > 0.001f) {
                v_sum.x += v.x; v_sum.y += v.y; v_sum.z += v.z;
                n_sum.x += n.x; n_sum.y += n.y; n_sum.z += n.z;
                count++;
            }
        }
    }

    int didx = y * dst_w + x;
    if (count > 0) {
        v_dst[didx] = make_float3(v_sum.x/count, v_sum.y/count, v_sum.z/count);
        float len = sqrtf(n_sum.x*n_sum.x + n_sum.y*n_sum.y + n_sum.z*n_sum.z);
        if (len > 1e-6f) {
            n_dst[didx] = make_float3(n_sum.x/len, n_sum.y/len, n_sum.z/len);
        } else {
            n_dst[didx] = make_float3(0,0,0);
        }
    } else {
        v_dst[didx] = make_float3(0,0,0);
        n_dst[didx] = make_float3(0,0,0);
    }
}

// ---------------------------------------------------------------------------
// Hessian Reduction Kernel
// Parallelized over live frame pixels
// ---------------------------------------------------------------------------
__global__ void computeHessianKernel(
    const float3* live_vertices,
    const float3* live_normals,
    const float3* model_vertices,
    const float3* model_normals,
    int width, int height,
    int model_w, int model_h,
    float fx, float fy, float cx, float cy,
    float dist_thresh, float angle_thresh_cos,
    // Relative transform: ref_cam <- live_cam
    float r00, float r01, float r02,
    float r10, float r11, float r12,
    float r20, float r21, float r22,
    float tx,  float ty,  float tz,
    // World to Ref transform (for model vertices/normals)
    float rw00, float rw01, float rw02,
    float rw10, float rw11, float rw12,
    float rw20, float rw21, float rw22,
    float twx,  float twy,  float twz,
    // Output: 34 floats
    float* global_stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float local_A[21] = {0};
    float local_b[6]  = {0};
    float local_res   = 0;
    int   local_inliers = 0;
    int   local_valid_live = 0;
    int   local_valid_model = 0;
    int   local_projected = 0;
    int   local_dist_filtered = 0;
    int   local_angle_filtered = 0;

    if (x < width && y < height) {
        int idx = y * width + x;
        float3 v_live = live_vertices[idx];
        
        if (v_live.z > 0.001f) {
            local_valid_live++;
            float3 v_ref;
            v_ref.x = r00 * v_live.x + r01 * v_live.y + r02 * v_live.z + tx;
            v_ref.y = r10 * v_live.x + r11 * v_live.y + r12 * v_live.z + ty;
            v_ref.z = r20 * v_live.x + r21 * v_live.y + r22 * v_live.z + tz;

            if (v_ref.z > 0.001f) {
                int mx = __float2int_rd(fx * v_ref.x / v_ref.z + cx + 0.5f);
                int my = __float2int_rd(fy * v_ref.y / v_ref.z + cy + 0.5f);

                if (mx >= 0 && mx < model_w && my >= 0 && my < model_h) {
                    local_projected++;
                    int midx = my * model_w + mx;
                    float3 v_model_world = model_vertices[midx];
                    float3 n_model_world = model_normals[midx];

                    // v_model_world is in WORLD space \u2014 z can be negative. Use norm check (zero = no hit).
                    float mv_norm_sq = v_model_world.x*v_model_world.x + v_model_world.y*v_model_world.y + v_model_world.z*v_model_world.z;
                    float mn_norm_sq = n_model_world.x*n_model_world.x + n_model_world.y*n_model_world.y + n_model_world.z*n_model_world.z;
                    if (mv_norm_sq > 1e-12f && mn_norm_sq > 0.9f) {
                        float3 v_model;
                        v_model.x = rw00 * v_model_world.x + rw01 * v_model_world.y + rw02 * v_model_world.z + twx;
                        v_model.y = rw10 * v_model_world.x + rw11 * v_model_world.y + rw12 * v_model_world.z + twy;
                        v_model.z = rw20 * v_model_world.x + rw21 * v_model_world.y + rw22 * v_model_world.z + twz;
                        
                        float3 n_model;
                        n_model.x = rw00 * n_model_world.x + rw01 * n_model_world.y + rw02 * n_model_world.z;
                        n_model.y = rw10 * n_model_world.x + rw11 * n_model_world.y + rw12 * n_model_world.z;
                        n_model.z = rw20 * n_model_world.x + rw21 * n_model_world.y + rw22 * n_model_world.z;

                        local_valid_model++;
                        float dx = v_ref.x - v_model.x;
                        float dy = v_ref.y - v_model.y;
                        float dz = v_ref.z - v_model.z;
                        float dist_sq = dx*dx + dy*dy + dz*dz;

                        if (dist_sq < dist_thresh * dist_thresh) {
                            float3 n_live = live_normals[idx];
                            float3 n_live_ref;
                            n_live_ref.x = r00 * n_live.x + r01 * n_live.y + r02 * n_live.z;
                            n_live_ref.y = r10 * n_live.x + r11 * n_live.y + r12 * n_live.z;
                            n_live_ref.z = r20 * n_live.x + r21 * n_live.y + r22 * n_live.z;

                            float dot = n_live_ref.x * n_model.x + n_live_ref.y * n_model.y + n_live_ref.z * n_model.z;
                            if (fabsf(dot) > angle_thresh_cos) {
                                float err = n_model.x * dx + n_model.y * dy + n_model.z * dz;
                                float3 n_model_live;
                                n_model_live.x = r00 * n_model.x + r10 * n_model.y + r20 * n_model.z;
                                n_model_live.y = r01 * n_model.x + r11 * n_model.y + r21 * n_model.z;
                                n_model_live.z = r02 * n_model.x + r12 * n_model.y + r22 * n_model.z;

                                float J[6];
                                J[0] = n_model_live.x; J[1] = n_model_live.y; J[2] = n_model_live.z;
                                J[3] = v_live.y * n_model_live.z - v_live.z * n_model_live.y;
                                J[4] = v_live.z * n_model_live.x - v_live.x * n_model_live.z;
                                J[5] = v_live.x * n_model_live.y - v_live.y * n_model_live.x;

                                // KIN-FORK: `continue` is invalid here (no enclosing loop —
                                // nvcc hard error). Guard form copied from ICPTracker_hip.hip:220.
                                if (!isnan(J[0]) && !isinf(J[0]) && !isnan(J[3]) && !isinf(J[3])) {
                                    float abs_err = fabsf(err);
                                    // Canonical Huber IRLS (tracking/ICPShared.h,
                                    // CPU ICPTracker.cpp): curvature w*J*J^T,
                                    // gradient J*(w*e), objective psi(|e|).
                                    // The unweighted curvature here made a
                                    // 3 deg / 4 cm motion diverge (GPU-02).
                                    const float huber_k = kHuberK;
                                    float w = (abs_err <= huber_k) ? 1.0f : huber_k / abs_err;
                                    float weighted_err = err * w;

                                    int count = 0;
                                    for (int i = 0; i < 6; ++i) {
                                        for (int j = i; j < 6; ++j) {
                                            local_A[count++] += w * J[i] * J[j];
                                        }
                                        local_b[i] -= J[i] * weighted_err;
                                    }
                                    local_res += (abs_err <= huber_k)
                                                     ? abs_err * abs_err
                                                     : 2.0f * huber_k * abs_err - huber_k * huber_k;
                                    local_inliers++;
                                }
                            } else local_angle_filtered++;
                        } else local_dist_filtered++;
                    }
                }
            }
        }
    }

    // Pack for warp reduction
    float vals[34];
    for (int i = 0; i < 21; ++i) vals[i] = local_A[i];
    for (int i = 0; i < 6; ++i)  vals[21 + i] = local_b[i];
    vals[27] = local_res;
    vals[28] = (float)local_inliers;
    vals[29] = (float)local_valid_live;
    vals[30] = (float)local_valid_model;
    vals[31] = (float)local_projected;
    vals[32] = (float)local_dist_filtered;
    vals[33] = (float)local_angle_filtered;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = (blockDim.x * blockDim.y) / 32;
    __shared__ float s_reduce[8][34];

    for (int i = 0; i < 34; ++i) {
        float val = vals[i];
        for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) s_reduce[wid][i] = val;
    }
    __syncthreads();
    if (tid < 34) {
        float sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) sum += s_reduce[w][tid];
        atomicAdd(&global_stats[tid], sum);
    }
}

// ---------------------------------------------------------------------------
// Host-side implementation
// ---------------------------------------------------------------------------

void ICPTracker::initGPU() {
    d_hessian_ = utils::make_cuda_unique<float>(34);
    for (int l = 0; l < sensor::FramePyramid::LEVELS; ++l) {
        int n = (sensor::FRAME_W >> l) * (sensor::FRAME_H >> l);
        d_pyramid_v[l] = utils::make_cuda_unique<float3>(n);
        d_pyramid_n[l] = utils::make_cuda_unique<float3>(n);
    }
}

void ICPTracker::freeGPU() {
    d_hessian_.reset();
    for (int l = 0; l < sensor::FramePyramid::LEVELS; ++l) {
        d_pyramid_v[l].reset();
        d_pyramid_n[l].reset();
    }
}

ICPResult ICPTracker::trackGPU(const float*                d_depth,
                               const uint8_t*              d_rgb,
                               int                         width,
                               int                         height,
                               const ModelFrame&           model,
                               const Eigen::Matrix4f&      pose_estimate,
                               const Eigen::Matrix4f&      ref_pose)
{
    ICPResult result;
    result.pose = pose_estimate;
    result.tracking_ok = true;

    // 1. Build Level 0 on GPU
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    computeVerticesKernel<<<grid, block>>>(d_depth, d_pyramid_v[0].get(), width, height, sensor::FX, sensor::FY, sensor::CX, sensor::CY);
    computeNormalsKernel<<<grid, block>>>(d_depth, d_pyramid_v[0].get(), d_pyramid_n[0].get(), width, height);

    // 2. Build Pyramid on GPU
    for (int l = 1; l < sensor::FramePyramid::LEVELS; ++l) {
        int sw = width >> (l - 1);
        int sh = height >> (l - 1);
        int dw = width >> l;
        int dh = height >> l;
        dim3 g((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
        downsampleKernel<<<g, block>>>(d_pyramid_v[l-1].get(), d_pyramid_n[l-1].get(), sw, sh, d_pyramid_v[l].get(), d_pyramid_n[l].get(), dw, dh);
    }
    cudaDeviceSynchronize();

    // 3. Track levels
    bool converged = false;
    for (int level = sensor::FramePyramid::LEVELS - 1; level >= 0; --level) {
        result = trackLevelGPU(
            d_pyramid_v[level].get(), d_pyramid_n[level].get(),
            width >> level, height >> level,
            model, result.pose, ref_pose, level, params_.max_iterations[level]
        );
        converged = result.converged;
    }
    result.tracking_ok = converged && result.inliers > 100;
    return result;
}

ICPResult ICPTracker::trackLevelGPU(const float3*            d_v_live,
                                   const float3*            d_n_live,
                                   int                      width,
                                   int                      height,
                                   const ModelFrame&        model,
                                   const Eigen::Matrix4f&   pose_estimate,
                                   const Eigen::Matrix4f&   ref_pose,
                                   int                      level,
                                   int                      max_iter)
{
    ICPResult result;
    result.pose = pose_estimate;

    // Intrinsics for projecting into the MODEL frame (which is always full resolution)
    float fx = kfusion::sensor::FX;
    float fy = kfusion::sensor::FY;
    float cx = kfusion::sensor::CX;
    float cy = kfusion::sensor::CY;

    float angle_thresh_cos = cosf(params_.angle_threshold * 3.14159f / 180.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        CUDA_CHECK(cudaMemset(d_hessian_.get(), 0, 34 * sizeof(float)));

        Eigen::Matrix4f rel = ref_pose.inverse() * result.pose;
        Eigen::Matrix3f R = rel.block<3,3>(0,0);
        Eigen::Vector3f t = rel.block<3,1>(0,3);

        Eigen::Matrix4f w2ref = ref_pose.inverse();
        Eigen::Matrix3f Rw = w2ref.block<3,3>(0,0);
        Eigen::Vector3f tw = w2ref.block<3,1>(0,3);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        computeHessianKernel<<<grid, block>>>(
            d_v_live, d_n_live, 
            model.d_vertices.get(), model.d_normals.get(),
            width, height,
            model.width, model.height,
            fx, fy, cx, cy,
            params_.dist_threshold, angle_thresh_cos,
            R(0,0), R(0,1), R(0,2),
            R(1,0), R(1,1), R(1,2),
            R(2,0), R(2,1), R(2,2),
            t.x(), t.y(), t.z(),
            Rw(0,0), Rw(0,1), Rw(0,2),
            Rw(1,0), Rw(1,1), Rw(1,2),
            Rw(2,0), Rw(2,1), Rw(2,2),
            tw.x(), tw.y(), tw.z(),
            d_hessian_.get()
        );
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_hessian[34];
        CUDA_CHECK(cudaMemcpy(h_hessian, d_hessian_.get(), 34 * sizeof(float), cudaMemcpyDeviceToHost));

        Eigen::Matrix<float,6,6> A = Eigen::Matrix<float,6,6>::Zero();
        Eigen::Matrix<float,6,1> b = Eigen::Matrix<float,6,1>::Zero();
        int count = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                A(i,j) = A(j,i) = h_hessian[count++];
            }
            b(i) = h_hessian[21 + i];
        }
        float residual = h_hessian[27];
        int inliers = (int)h_hessian[28];
        
        // Populate diagnostics
        result.valid_live_points  = (int)h_hessian[29];
        result.valid_model_points = (int)h_hessian[30];
        result.projected_points   = (int)h_hessian[31];
        result.dist_filtered      = (int)h_hessian[32];
        result.angle_filtered     = (int)h_hessian[33];

        if (inliers < 10) break;

        // Solve update with adaptive Tikhonov regularization (Levenberg-Marquardt style)
        float damping = 0.1f;
        if (level == 0) damping = 0.01f; // Finer levels need less damping if close to convergence

        // Numerical Integrity Check: Check Hessian condition number
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> es(A);
        float min_eig = es.eigenvalues().minCoeff();
        float max_eig = es.eigenvalues().maxCoeff();
        float cond = (min_eig > 0) ? (max_eig / min_eig) : 1e10f;

        // If system is extremely ill-conditioned, tracking is unreliable (e.g. flat wall)
        if (cond > 1e7f || min_eig < 1e-4f) {
            damping = 1.0f; // Increase damping to stabilize
        }

        A += Eigen::Matrix<float,6,6>::Identity() * damping;
        Eigen::Matrix<float,6,1> update = A.ldlt().solve(b);

        // Reject NaNs or catastrophically large translational updates (>20cm per iteration)
        if (!update.allFinite() || update.segment<3>(0).norm() > 0.2f) {
            break;
        }

        // Robustness: Detect erratic updates (too much rotation in one step)
        if (update.segment<3>(3).norm() > 0.5f) { // ~30 degrees per iteration
            break;
        }

        // Update pose (Rodrigues approximation)
        float rot_norm = update.segment<3>(3).norm();
        Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
        if (rot_norm > 1e-6f) {
            dR = Eigen::AngleAxisf(rot_norm, update.segment<3>(3).normalized()).toRotationMatrix();
        } else {
            dR(0,1) = -update(5); dR(0,2) =  update(4);
            dR(1,0) =  update(5); dR(1,2) = -update(3);
            dR(2,0) = -update(4); dR(2,1) =  update(3);
        }

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3,3>(0,0) = dR;
        delta.block<3,1>(0,3) = update.head<3>();

        // IMPORTANT: The twist (update) was calculated in the LIVE camera frame.
        // Therefore, we must apply it by right-multiplying the delta transform.
        result.pose = result.pose * delta;

        // Orthonormalize rotation to prevent drift/ghosting
        Eigen::Matrix3f R_curr = result.pose.block<3,3>(0,0);
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(R_curr, Eigen::ComputeFullU | Eigen::ComputeFullV);
        result.pose.block<3,3>(0,0) = svd.matrixU() * svd.matrixV().transpose();
        result.inliers = inliers;
        result.error = residual / fmaxf(1.0f, (float)inliers);

        if (update.norm() < 5e-5f) {
            result.converged = true;
            break;
        }
    }

    return result;
}

} // namespace tracking
} // namespace kfusion

#endif // CUDA_ENABLED
