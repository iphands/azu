#ifdef HIP_ENABLED

#include "tsdf/TSDFVolume.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstring>
#include "tsdf/VoxelGPU.h"

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        } \
    } while(0)

#define HIP_CHECK_LAST() \
    do { \
        hipError_t err = hipGetLastError(); \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        } \
    } while(0)

namespace kfusion {
namespace tsdf {

// -------------------------------------------------------------------
// Integration kernel: Image-Centric (Pixel Parallel)
// -------------------------------------------------------------------
__global__ void integrationKernel_VoxelParallel(
    VoxelGPU* voxels, int resolution, float voxel_size, float3 origin, float truncation, float max_weight,
    const float* depth, const uint8_t* rgb, int width, int height,
    float fx, float fy, float cx, float cy,
    float rwc00, float rwc01, float rwc02, float rwc10, float rwc11, float rwc12, float rwc20, float rwc21, float rwc22,
    float twc_x, float twc_y, float twc_z,
    float min_depth, float max_depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution || y >= resolution || z >= resolution) return;

    int vidx = z * resolution * resolution + y * resolution + x;
    VoxelGPU& vox = voxels[vidx];

    float3 world_pos = {
        origin.x + x * voxel_size,
        origin.y + y * voxel_size,
        origin.z + z * voxel_size
    };

    // Project world point back into camera
    float cx_c = rwc00 * world_pos.x + rwc01 * world_pos.y + rwc02 * world_pos.z + twc_x;
    float cy_c = rwc10 * world_pos.x + rwc11 * world_pos.y + rwc12 * world_pos.z + twc_y;
    float cz_c = rwc20 * world_pos.x + rwc21 * world_pos.y + rwc22 * world_pos.z + twc_z;

    if (cz_c <= 0.1f) return;

    int u = __float2int_rn(fx * cx_c / cz_c + cx);
    int v = __float2int_rn(fy * cy_c / cz_c + cy);

    if (u < 0 || u >= width || v < 0 || v >= height) return;

    float d_meas = depth[v * width + u];
    // Filter invalid depths: too close, too far, or NaN/inf
    if (d_meas < 0.1f || d_meas > max_depth || isnan(d_meas) || isinf(d_meas)) return;

    float sdf = d_meas - cz_c;
    if (sdf < -truncation) return;
    if (sdf > truncation) return;

    float tsdf_new = fminf(1.0f, sdf / truncation);
    if (isnan(tsdf_new) || isinf(tsdf_new)) return;
    
    float w_old = vox.weight;
    float w_new = 1.0f;
    float w_sum = fminf(w_old + w_new, max_weight);

    float new_tsdf = (vox.tsdf * w_old + tsdf_new * w_new) / (w_old + w_new + 1e-6f);
    if (isnan(new_tsdf) || isinf(new_tsdf)) return;

    vox.tsdf   = new_tsdf;
    vox.weight = w_sum;

    if (rgb && sdf > -truncation * 0.5f) {
        int pidx = (v * width + u) * 3;
        // Store colors directly in 0-255 range (no normalization)
        float r_meas = (float)rgb[pidx + 0];
        float g_meas = (float)rgb[pidx + 1];
        float b_meas = (float)rgb[pidx + 2];
        
        vox.r = (vox.r * w_old + r_meas) / (w_old + 1.0f + 1e-6f);
        vox.g = (vox.g * w_old + g_meas) / (w_old + 1.0f + 1e-6f);
        vox.b = (vox.b * w_old + b_meas) / (w_old + 1.0f + 1e-6f);
    }
}

// -------------------------------------------------------------------
// Host-side GPU management
// -------------------------------------------------------------------
void TSDFVolume::freeGPU() {
    d_voxels_.reset();
    d_pc_is_valid_.reset();
    d_pc_offsets_.reset();
    // KIN-FORK: points/colors must die with the rest — extractGlobalPointCloudGPU
    // reallocs via if(!ptr), so surviving old-size buffers mean OOB writes on grow.
    d_pc_out_points_.reset();
    d_pc_out_colors_.reset();
    gpu_valid_ = false;
}

void TSDFVolume::initGPU() {
    // Idempotent: an allocated device volume IS the scan (the host copy is stale
    // while the GPU integrates), so a restart must not re-upload over it.
    // freeGPU() (reset, resolution change) is what forces a fresh upload.
    if (gpu_valid_ && d_voxels_) return;
    const int res = params_.resolution;
    size_t n = static_cast<size_t>(res) * static_cast<size_t>(res) * static_cast<size_t>(res);
    d_voxels_ = utils::make_hip_unique<VoxelGPU>(n);
    gpu_valid_ = true;
    syncToGPU();
}

void TSDFVolume::syncToGPU() {
    if (!gpu_valid_) return;
    size_t n = voxels_.size();
    const size_t CHUNK_SIZE = 1024 * 1024;
    std::vector<VoxelGPU> gpu_data(std::min(n, CHUNK_SIZE));
    for (size_t offset = 0; offset < n; offset += CHUNK_SIZE) {
        size_t current_chunk = std::min(CHUNK_SIZE, n - offset);
        for (size_t i = 0; i < current_chunk; ++i) {
            size_t idx = offset + i;
            gpu_data[i].tsdf   = voxels_[idx].tsdf;
            gpu_data[i].weight = voxels_[idx].weight;
            // Store colors directly in 0-255 range (no normalization)
            gpu_data[i].r      = static_cast<float>(voxels_[idx].r);
            gpu_data[i].g      = static_cast<float>(voxels_[idx].g);
            gpu_data[i].b      = static_cast<float>(voxels_[idx].b);
        }
        HIP_CHECK(hipMemcpy(d_voxels_.get() + offset, gpu_data.data(), current_chunk * sizeof(VoxelGPU), hipMemcpyHostToDevice));
    }
}

void TSDFVolume::syncFromGPU() {
    if (!gpu_valid_) return;
    size_t n = voxels_.size();
    const size_t CHUNK_SIZE = 1024 * 1024;
    std::vector<VoxelGPU> gpu_data(std::min(n, CHUNK_SIZE));
    for (size_t offset = 0; offset < n; offset += CHUNK_SIZE) {
        size_t current_chunk = std::min(CHUNK_SIZE, n - offset);
        HIP_CHECK(hipMemcpy(gpu_data.data(), d_voxels_.get() + offset, current_chunk * sizeof(VoxelGPU), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < current_chunk; ++i) {
            size_t idx = offset + i;
            float w = gpu_data[i].weight;
            voxels_[idx].weight = w;
            if (w > 0.001f) {
                voxels_[idx].tsdf = gpu_data[i].tsdf;
                // Store colors directly in 0-255 range (no denormalization)
                voxels_[idx].r = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].r));
                voxels_[idx].g = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].g));
                voxels_[idx].b = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].b));
            } else {
                voxels_[idx].tsdf = 1.0f;
            }
        }
    }
}

void TSDFVolume::integrateGPU(const float*           d_depth,
                               const uint8_t*         d_rgb,
                               const Eigen::Matrix4f& pose,
                               float fx, float fy,
                               float cx, float cy,
                               int   width, int height,
                               float min_depth,
                               float max_depth)
{
    if (!gpu_valid_) return;

    Eigen::Matrix3f R_cw = pose.block<3,3>(0,0);
    Eigen::Vector3f t_cw = pose.block<3,1>(0,3);
    Eigen::Matrix4f inv  = pose.inverse();
    Eigen::Matrix3f R_wc = inv.block<3,3>(0,0);
    Eigen::Vector3f t_wc = inv.block<3,1>(0,3);

    float3 origin = {params_.origin.x(), params_.origin.y(), params_.origin.z()};
    
    // Launch one thread per voxel
    dim3 block(8, 8, 8);
    int res = params_.resolution;
    dim3 grid((res + block.x - 1) / block.x, (res + block.y - 1) / block.y, (res + block.z - 1) / block.z);

    integrationKernel_VoxelParallel<<<grid, block>>>(
        (VoxelGPU*)d_voxels_.get(), params_.resolution, params_.voxel_size, origin, params_.truncation, params_.max_weight,
        d_depth, d_rgb, width, height, fx, fy, cx, cy,
        R_wc(0,0), R_wc(0,1), R_wc(0,2), R_wc(1,0), R_wc(1,1), R_wc(1,2), R_wc(2,0), R_wc(2,1), R_wc(2,2),
        t_wc.x(), t_wc.y(), t_wc.z(),
        min_depth, max_depth
    );
    HIP_CHECK_LAST();
    HIP_CHECK(hipDeviceSynchronize());
}

__device__ float get_tsdf_trilinear(VoxelGPU* voxels, int resolution, float voxel_size, float3 origin, float3 p) {
    float3 v = make_float3((p.x - origin.x) / voxel_size, (p.y - origin.y) / voxel_size, (p.z - origin.z) / voxel_size);
    
    int x0 = __float2int_rd(v.x);
    int y0 = __float2int_rd(v.y);
    int z0 = __float2int_rd(v.z);

    if (x0 < 0 || x0 >= resolution - 1 || y0 < 0 || y0 >= resolution - 1 || z0 < 0 || z0 >= resolution - 1) {
        if (x0 >= 0 && x0 < resolution && y0 >= 0 && y0 < resolution && z0 >= 0 && z0 < resolution) {
            VoxelGPU& vox = voxels[z0 * resolution * resolution + y0 * resolution + x0];
            return (vox.weight > 0.0f) ? vox.tsdf : 1.0f;
        }
        return 1.0f;
    }

    float tx = v.x - x0;
    float ty = v.y - y0;
    float tz = v.z - z0;

    auto getV = [&](int x, int y, int z) {
        VoxelGPU& vox = voxels[z * resolution * resolution + y * resolution + x];
        return (vox.weight > 0.0f) ? vox.tsdf : 1.0f;
    };

    float v000 = getV(x0, y0, z0);
    float v100 = getV(x0+1, y0, z0);
    float v010 = getV(x0, y0+1, z0);
    float v110 = getV(x0+1, y0+1, z0);
    float v001 = getV(x0, y0, z0+1);
    float v101 = getV(x0+1, y0, z0+1);
    float v011 = getV(x0, y0+1, z0+1);
    float v111 = getV(x0+1, y0+1, z0+1);

    float v00 = v000 * (1 - tx) + v100 * tx;
    float v01 = v001 * (1 - tx) + v101 * tx;
    float v10 = v010 * (1 - tx) + v110 * tx;
    float v11 = v011 * (1 - tx) + v111 * tx;

    float v0 = v00 * (1 - ty) + v10 * ty;
    float v1 = v01 * (1 - ty) + v11 * ty;

    return v0 * (1 - tz) + v1 * tz;
}

__device__ float3 computeNormalGPU(void* voxels_void, int resolution, float voxel_size, float3 origin, float3 p) {
    VoxelGPU* voxels = (VoxelGPU*)voxels_void;

    float3 n;
    n.x = get_tsdf_trilinear(voxels, resolution, voxel_size, origin, make_float3(p.x + voxel_size, p.y, p.z)) -
          get_tsdf_trilinear(voxels, resolution, voxel_size, origin, make_float3(p.x - voxel_size, p.y, p.z));
    n.y = get_tsdf_trilinear(voxels, resolution, voxel_size, origin, make_float3(p.x, p.y + voxel_size, p.z)) -
          get_tsdf_trilinear(voxels, resolution, voxel_size, origin, make_float3(p.x, p.y - voxel_size, p.z));
    n.z = get_tsdf_trilinear(voxels, resolution, voxel_size, origin, make_float3(p.x, p.y, p.z + voxel_size)) -
          get_tsdf_trilinear(voxels, resolution, voxel_size, origin, make_float3(p.x, p.y, p.z - voxel_size));

    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-6f) {
        n.x /= len; n.y /= len; n.z /= len;
    } else {
        n = make_float3(0, 0, 0);
    }
    return n;
}

__global__ void raycastKernel(
    void* voxels_void, int resolution, float voxel_size, float3 origin,
    float fx, float fy, float cx, float cy,
    float r00, float r01, float r02, float tx,
    float r10, float r11, float r12, float ty,
    float r20, float r21, float r22, float tz,
    int width, int height,
    float3* out_v, float3* out_n, uchar3* out_c)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    VoxelGPU* voxels = (VoxelGPU*)voxels_void;



    // Ray in camera space
    float3 ray_c = make_float3((px - cx) / fx, (py - cy) / fy, 1.0f);
    float rlen = sqrtf(ray_c.x*ray_c.x + ray_c.y*ray_c.y + 1.0f);
    ray_c.x /= rlen; ray_c.y /= rlen; ray_c.z /= rlen;

    // Ray in world space
    float3 ray_w;
    ray_w.x = r00 * ray_c.x + r01 * ray_c.y + r02 * ray_c.z;
    ray_w.y = r10 * ray_c.x + r11 * ray_c.y + r12 * ray_c.z;
    ray_w.z = r20 * ray_c.x + r21 * ray_c.y + r22 * ray_c.z;

    float3 pos_w = make_float3(tx, ty, tz);
    float t = 0.3f; // min depth for Kinect v1
    float prev_tsdf = 1.0f;
    float step = voxel_size;

    while (t < 5.0f) {
        float3 p = make_float3(pos_w.x + ray_w.x * t, pos_w.y + ray_w.y * t, pos_w.z + ray_w.z * t);
        float tsdf = get_tsdf_trilinear(voxels, resolution, voxel_size, origin, p);

        if (tsdf < 1.0f) {
            // Probable geometry region — only detect crossings here, matching CPU raycast.
            if (prev_tsdf > 0.0f && tsdf <= 0.0f) {
                // Surface zero-crossing found
                float t_hit = t - step * tsdf / (tsdf - prev_tsdf + 1e-6f);
                if (isnan(t_hit) || isinf(t_hit)) {
                    out_v[py * width + px] = make_float3(0,0,0);
                    return;
                }
                float3 pf = make_float3(pos_w.x + ray_w.x * t_hit, pos_w.y + ray_w.y * t_hit, pos_w.z + ray_w.z * t_hit);

                if (isnan(pf.x) || isnan(pf.y) || isnan(pf.z)) {
                    out_v[py * width + px] = make_float3(0,0,0);
                    return;
                }

                out_v[py * width + px] = pf;
                out_n[py * width + px] = computeNormalGPU(voxels_void, resolution, voxel_size, origin, pf);

                if (out_c) {
                    int vx = __float2int_rd((pf.x - origin.x) / voxel_size);
                    int vy = __float2int_rd((pf.y - origin.y) / voxel_size);
                    int vz = __float2int_rd((pf.z - origin.z) / voxel_size);
                    if (vx >= 0 && vx < resolution && vy >= 0 && vy < resolution && vz >= 0 && vz < resolution) {
                        VoxelGPU& v = voxels[vz * resolution * resolution + vy * resolution + vx];
                        // Store colors directly in 0-255 range (no denormalization)
                        out_c[py * width + px] = make_uchar3(
                            (uint8_t)fminf(255.0f, fmaxf(0.0f, v.r)),
                            (uint8_t)fminf(255.0f, fmaxf(0.0f, v.g)),
                            (uint8_t)fminf(255.0f, fmaxf(0.0f, v.b))
                        );
                    }
                }
                return;
            }
            prev_tsdf = tsdf;
            step = voxel_size * 0.5f; // Small steps near surface
        } else {
            // Unknown or empty space — reset prev_tsdf so a one-step skip from empty
            // into negative TSDF does not fire a false crossing, matching CPU behaviour.
            prev_tsdf = 1.0f;
            step = voxel_size;
        }
        t += step;
    }
    
    out_v[py * width + px] = make_float3(0,0,0);
    out_n[py * width + px] = make_float3(0,0,0);
    if (out_c) out_c[py * width + px] = make_uchar3(0,0,0);
}

void TSDFVolume::raycastGPU(const Eigen::Matrix4f& pose,
                             float fx, float fy, float cx, float cy,
                             int width, int height,
                             float3* d_vertices, float3* d_normals,
                             uchar3* d_colors)
{
    if (!gpu_valid_) return;

    Eigen::Matrix3f R = pose.block<3,3>(0,0);
    Eigen::Vector3f t = pose.block<3,1>(0,3);

    float3 f_origin = {params_.origin.x(), params_.origin.y(), params_.origin.z()};
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    raycastKernel<<<grid, block>>>(
        d_voxels_.get(), params_.resolution, params_.voxel_size, f_origin,
        fx, fy, cx, cy,
        R(0,0), R(0,1), R(0,2), t.x(),
        R(1,0), R(1,1), R(1,2), t.y(),
        R(2,0), R(2,1), R(2,2), t.z(),
        width, height, d_vertices, d_normals, d_colors
    );
    HIP_CHECK_LAST();
    HIP_CHECK(hipDeviceSynchronize());
}

// -------------------------------------------------------------------
// Global Point Cloud Extraction (GPU-accelerated)
// -------------------------------------------------------------------

__global__ void classifyVoxelKernel(
    const VoxelGPU* voxels,
    int             resolution,
    uint32_t*       is_valid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution || y >= resolution || z >= resolution) return;

    int idx = z * resolution * resolution + y * resolution + x;
    const VoxelGPU& v = voxels[idx];

    // Same criteria as CPU extractGlobalPointCloud: weight > 1.0 and surface proximity
    if (v.weight > 1.0f && fabsf(v.tsdf) < 0.2f) {
        is_valid[idx] = 1;
    } else {
        is_valid[idx] = 0;
    }
}

__global__ void compactPointsKernel(
    const VoxelGPU* voxels,
    const uint32_t* is_valid,
    const uint32_t* offsets,
    int             resolution,
    float           voxel_size,
    float3          origin,
    float3*         out_points,
    uchar3*         out_colors)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution || y >= resolution || z >= resolution) return;

    int idx = z * resolution * resolution + y * resolution + x;
    if (is_valid[idx]) {
        uint32_t out_idx = offsets[idx];
        out_points[out_idx] = make_float3(
            origin.x + x * voxel_size,
            origin.y + y * voxel_size,
            origin.z + z * voxel_size
        );
        const VoxelGPU& v = voxels[idx];
        // Store colors directly in 0-255 range (no denormalization)
        out_colors[out_idx] = make_uchar3(
            (uint8_t)fmaxf(0, fminf(255, v.r)),
            (uint8_t)fmaxf(0, fminf(255, v.g)),
            (uint8_t)fmaxf(0, fminf(255, v.b))
        );
    }
}

void TSDFVolume::extractGlobalPointCloudGPU(std::vector<Eigen::Vector3f>& points_out,
                                            std::vector<uint8_t>&         colors_out) const
{
    int res = params_.resolution;
    int total_voxels = res * res * res;

    if (!d_pc_is_valid_) d_pc_is_valid_ = utils::make_hip_unique<uint32_t>(total_voxels);
    if (!d_pc_offsets_) d_pc_offsets_ = utils::make_hip_unique<uint32_t>(total_voxels);
    if (!d_pc_out_points_) d_pc_out_points_ = utils::make_hip_unique<float3>(total_voxels);
    if (!d_pc_out_colors_) d_pc_out_colors_ = utils::make_hip_unique<uchar3>(total_voxels);

    dim3 block(8, 8, 8);
    dim3 grid((res + block.x - 1) / block.x, (res + block.y - 1) / block.y, (res + block.z - 1) / block.z);

    classifyVoxelKernel<<<grid, block>>>(d_voxels_.get(), res, d_pc_is_valid_.get());
    HIP_CHECK_LAST();
    HIP_CHECK(hipDeviceSynchronize());

    // Host-side exclusive prefix scan (no rocThrust dependency)
    std::vector<uint32_t> h_valid(total_voxels);
    HIP_CHECK(hipMemcpy(h_valid.data(), d_pc_is_valid_.get(), total_voxels * sizeof(uint32_t), hipMemcpyDeviceToHost));

    std::vector<uint32_t> h_offsets(total_voxels);
    uint32_t running = 0;
    for (int i = 0; i < total_voxels; ++i) {
        h_offsets[i] = running;
        running += h_valid[i];
    }
    uint32_t total_points = running;

    if (total_points == 0) return;

    HIP_CHECK(hipMemcpy(d_pc_offsets_.get(), h_offsets.data(), total_voxels * sizeof(uint32_t), hipMemcpyHostToDevice));

    compactPointsKernel<<<grid, block>>>(
        d_voxels_.get(), d_pc_is_valid_.get(), d_pc_offsets_.get(),
        res, params_.voxel_size,
        make_float3(params_.origin.x(), params_.origin.y(), params_.origin.z()),
        d_pc_out_points_.get(), d_pc_out_colors_.get()
    );
    HIP_CHECK_LAST();
    HIP_CHECK(hipDeviceSynchronize());

    points_out.resize(total_points);
    colors_out.resize(total_points * 3);

    // Direct copy from device into the correctly sized Eigen/byte vectors
    HIP_CHECK(hipMemcpy(points_out.data(), d_pc_out_points_.get(), total_points * sizeof(float3), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(colors_out.data(), d_pc_out_colors_.get(), total_points * sizeof(uchar3), hipMemcpyDeviceToHost));
}

} // namespace tsdf
} // namespace kfusion

#endif // HIP_ENABLED
