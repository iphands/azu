#ifdef CUDA_ENABLED

#include "tsdf/TSDFVolume.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <iostream>
#include <cstring>
#include "tsdf/VoxelGPU.h"

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
namespace tsdf {

// -------------------------------------------------------------------
// Integration kernel: voxel-parallel (one thread per voxel)
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

    const size_t vidx = (static_cast<size_t>(z) * resolution + y) * resolution + x;
    VoxelGPU& vox = voxels[vidx];

    // CPU canonical integration (src/tsdf/TSDFVolume.cpp integrateCPU,
    // docs/CANONICAL_SEMANTICS.md): the voxel CORNER origin + i*vs projects to
    // pixel floor(p + 0.5); the measured depth must lie in the configured band;
    // free space in front of the surface is observed too (tsdf 1), which the
    // raycast's observed-only sampling relies on; colour within |sdf| < trunc/2.
    const float3 world_pos = {
        origin.x + x * voxel_size,
        origin.y + y * voxel_size,
        origin.z + z * voxel_size
    };
    const float cx_c = rwc00 * world_pos.x + rwc01 * world_pos.y + rwc02 * world_pos.z + twc_x;
    const float cy_c = rwc10 * world_pos.x + rwc11 * world_pos.y + rwc12 * world_pos.z + twc_y;
    const float cz_c = rwc20 * world_pos.x + rwc21 * world_pos.y + rwc22 * world_pos.z + twc_z;
    if (cz_c <= 0.0f) return;

    const float uf = floorf(fx * cx_c / cz_c + cx + 0.5f);
    const float vf = floorf(fy * cy_c / cz_c + cy + 0.5f);
    if (!(uf >= 0.0f && uf < (float)width && vf >= 0.0f && vf < (float)height)) return;
    const int pix = (int)vf * width + (int)uf;

    const float d_meas = depth[pix];
    if (!(d_meas >= min_depth && d_meas <= max_depth)) return;   // also rejects NaN

    const float sdf = d_meas - cz_c;
    if (sdf < -truncation) return;
    const float tsdf_new = fminf(1.0f, sdf / truncation);

    const float w_old = vox.weight;
    vox.tsdf   = (vox.tsdf * w_old + tsdf_new) / (w_old + 1.0f);
    vox.weight = fminf(w_old + 1.0f, max_weight);

    if (rgb && fabsf(sdf) < 0.5f * truncation) {
        // Device colour is sRGB [0,255] (host: [0,1]).
        const uint8_t* px = rgb + pix * 3;
        const float inv = 1.0f / (w_old + 1.0f);
        vox.r = (vox.r * w_old + (float)px[0]) * inv;
        vox.g = (vox.g * w_old + (float)px[1]) * inv;
        vox.b = (vox.b * w_old + (float)px[2]) * inv;
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
    d_voxels_ = utils::make_cuda_unique<VoxelGPU>(n);
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
            // Host colour is float sRGB [0,1]; device colour is sRGB [0,255].
            gpu_data[i].r      = voxels_[idx].r * 255.0f;
            gpu_data[i].g      = voxels_[idx].g * 255.0f;
            gpu_data[i].b      = voxels_[idx].b * 255.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_voxels_.get() + offset, gpu_data.data(), current_chunk * sizeof(VoxelGPU), cudaMemcpyHostToDevice));
    }
}

void TSDFVolume::syncFromGPU() {
    if (!gpu_valid_) return;
    size_t n = voxels_.size();
    const size_t CHUNK_SIZE = 1024 * 1024;
    std::vector<VoxelGPU> gpu_data(std::min(n, CHUNK_SIZE));
    for (size_t offset = 0; offset < n; offset += CHUNK_SIZE) {
        size_t current_chunk = std::min(CHUNK_SIZE, n - offset);
        CUDA_CHECK(cudaMemcpy(gpu_data.data(), d_voxels_.get() + offset, current_chunk * sizeof(VoxelGPU), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < current_chunk; ++i) {
            size_t idx = offset + i;
            float w = gpu_data[i].weight;
            voxels_[idx].weight = w;
            if (w > 0.001f) {
                voxels_[idx].tsdf = gpu_data[i].tsdf;
                // Device sRGB [0,255] -> host float sRGB [0,1] (the old code
                // stored a truncated byte value into the [0,1] float domain).
                voxels_[idx].r = fminf(255.0f, fmaxf(0.0f, gpu_data[i].r)) / 255.0f;
                voxels_[idx].g = fminf(255.0f, fmaxf(0.0f, gpu_data[i].g)) / 255.0f;
                voxels_[idx].b = fminf(255.0f, fmaxf(0.0f, gpu_data[i].b)) / 255.0f;
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
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Trilinear sample valid only when all 8 cell corners are observed (weight > 0),
// as the CPU sampleTSDF(): blending observed values with the unobserved +1
// sentinel manufactures zero crossings, i.e. phantom surfaces.
__device__ bool sample_observed(const VoxelGPU* voxels, int res, float vs, float3 origin,
                                float3 p, float* f_out) {
    const float vx = (p.x - origin.x) / vs, vy = (p.y - origin.y) / vs, vz = (p.z - origin.z) / vs;
    const float fx0 = floorf(vx), fy0 = floorf(vy), fz0 = floorf(vz);
    if (!(fx0 >= 0.0f && fy0 >= 0.0f && fz0 >= 0.0f &&
          fx0 < (float)(res - 1) && fy0 < (float)(res - 1) && fz0 < (float)(res - 1))) {
        return false;
    }
    const int x0 = (int)fx0, y0 = (int)fy0, z0 = (int)fz0;
    const float tx = vx - fx0, ty = vy - fy0, tz = vz - fz0;
    const size_t sy = (size_t)res, sz = (size_t)res * res;
    const VoxelGPU* b = voxels + (size_t)z0 * sz + (size_t)y0 * sy + x0;
    const VoxelGPU* c[8] = {b, b + 1, b + sy, b + sy + 1, b + sz, b + sz + 1, b + sz + sy, b + sz + sy + 1};
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (!(c[k]->weight > 0.0f)) return false;
    }
    const float v00 = c[0]->tsdf * (1 - tx) + c[1]->tsdf * tx;
    const float v10 = c[2]->tsdf * (1 - tx) + c[3]->tsdf * tx;
    const float v01 = c[4]->tsdf * (1 - tx) + c[5]->tsdf * tx;
    const float v11 = c[6]->tsdf * (1 - tx) + c[7]->tsdf * tx;
    const float v0 = v00 * (1 - ty) + v10 * ty;
    const float v1 = v01 * (1 - ty) + v11 * ty;
    *f_out = v0 * (1 - tz) + v1 * tz;
    return true;
}

__device__ bool normal_observed(const VoxelGPU* voxels, int res, float vs, float3 origin,
                                float3 p, float3* n_out) {
    float s[6];
    const float3 q[6] = {make_float3(p.x + vs, p.y, p.z), make_float3(p.x - vs, p.y, p.z),
                         make_float3(p.x, p.y + vs, p.z), make_float3(p.x, p.y - vs, p.z),
                         make_float3(p.x, p.y, p.z + vs), make_float3(p.x, p.y, p.z - vs)};
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        if (!sample_observed(voxels, res, vs, origin, q[k], &s[k])) return false;
    }
    const float3 n = make_float3(s[0] - s[1], s[2] - s[3], s[4] - s[5]);
    const float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (!(len > 1e-6f) || !isfinite(len)) return false;
    *n_out = make_float3(n.x / len, n.y / len, n.z / len);
    return true;
}

// CPU canonical raycast (src/tsdf/TSDFVolume.cpp raycast): slab clip to the
// volume inside the Z-depth band, adaptive step, FRONT faces only (first +->-
// bracket between observed samples, one secant step), normals from observed
// samples only. Every output is written on every path.
__global__ void raycastKernel(
    void* voxels_void, int resolution, float voxel_size, float3 origin,
    float truncation, float min_depth, float max_depth,
    float fx, float fy, float cx, float cy,
    float r00, float r01, float r02, float tx,
    float r10, float r11, float r12, float ty,
    float r20, float r21, float r22, float tz,
    int width, int height,
    float3* out_v, float3* out_n, uchar3* out_c)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    const int out_idx = py * width + px;
    out_v[out_idx] = make_float3(0, 0, 0);
    out_n[out_idx] = make_float3(0, 0, 0);
    if (out_c) out_c[out_idx] = make_uchar3(0, 0, 0);

    const VoxelGPU* voxels = (const VoxelGPU*)voxels_void;

    const float3 rc = make_float3((px - cx) / fx, (py - cy) / fy, 1.0f);
    const float rnorm = sqrtf(rc.x * rc.x + rc.y * rc.y + 1.0f);
    const float3 dir = make_float3((r00 * rc.x + r01 * rc.y + r02) / rnorm,
                                   (r10 * rc.x + r11 * rc.y + r12) / rnorm,
                                   (r20 * rc.x + r21 * rc.y + r22) / rnorm);
    const float3 o = make_float3(tx, ty, tz);

    float t_near = min_depth * rnorm;
    float t_far  = max_depth * rnorm;
    const float lo[3] = {origin.x, origin.y, origin.z};
    const float ext = (float)(resolution - 1) * voxel_size;
    const float oo[3] = {o.x, o.y, o.z};
    const float dd[3] = {dir.x, dir.y, dir.z};
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        const float inv = 1.0f / dd[a];
        float t0 = (lo[a] - oo[a]) * inv;
        float t1 = (lo[a] + ext - oo[a]) * inv;
        if (t0 > t1) { const float tmp = t0; t0 = t1; t1 = tmp; }
        if (isnan(t0) || isnan(t1)) {
            if (oo[a] < lo[a] || oo[a] > lo[a] + ext) t_far = -1.0f;
            continue;
        }
        t_near = fmaxf(t_near, t0);
        t_far  = fminf(t_far, t1);
    }
    if (!(t_near < t_far)) return;

    const float min_step = 0.5f * voxel_size;
    const float far_step = fmaxf(min_step, 0.8f * truncation);

    float t_prev = t_near;
    float f_prev = 1.0f;
    bool prev_valid = sample_observed(voxels, resolution, voxel_size, origin,
                                      make_float3(o.x + dir.x * t_prev, o.y + dir.y * t_prev, o.z + dir.z * t_prev),
                                      &f_prev);
    if (prev_valid && (!isfinite(f_prev) || f_prev < 0.0f)) return;

    float t_hit = 0.0f;
    bool found = false;
    for (int guard = 0; guard < 8192 && t_prev < t_far; ++guard) {
        const float step = (!prev_valid || f_prev >= 0.999f) ? far_step
                                                             : fmaxf(min_step, 0.8f * f_prev * truncation);
        const float t = fminf(t_prev + step, t_far);
        float f = 1.0f;
        const bool valid = sample_observed(voxels, resolution, voxel_size, origin,
                                           make_float3(o.x + dir.x * t, o.y + dir.y * t, o.z + dir.z * t), &f);
        if (valid && !isfinite(f)) return;
        if (valid && prev_valid) {
            if (f_prev > 0.0f && f <= 0.0f) {
                t_hit = t_prev + (t - t_prev) * (f_prev / (f_prev - f));
                found = true;
                break;
            }
            if (f_prev < 0.0f && f >= 0.0f) return;   // exiting a back face
        } else if (valid && f < 0.0f) {
            return;                                    // unobserved -> inside material
        }
        t_prev = t;
        f_prev = f;
        prev_valid = valid;
        if (t >= t_far) break;
    }
    if (!found) return;

    const float3 hit = make_float3(o.x + dir.x * t_hit, o.y + dir.y * t_hit, o.z + dir.z * t_hit);
    float3 n;
    if (!normal_observed(voxels, resolution, voxel_size, origin, hit, &n)) return;
    if (!isfinite(hit.x) || !isfinite(hit.y) || !isfinite(hit.z)) return;

    uchar3 col = make_uchar3(0, 0, 0);
    if (out_c) {
        const float vxf = floorf((hit.x - origin.x) / voxel_size);
        const float vyf = floorf((hit.y - origin.y) / voxel_size);
        const float vzf = floorf((hit.z - origin.z) / voxel_size);
        if (vxf >= 0.0f && vyf >= 0.0f && vzf >= 0.0f && vxf < (float)resolution &&
            vyf < (float)resolution && vzf < (float)resolution) {
            const VoxelGPU& v = voxels[((size_t)vzf * resolution + (size_t)vyf) * resolution + (size_t)vxf];
            if (v.weight > 0.0f) {
                if (!isfinite(v.r) || !isfinite(v.g) || !isfinite(v.b)) return;
                col = make_uchar3((uint8_t)__float2int_rn(fminf(255.0f, fmaxf(0.0f, v.r))),
                                  (uint8_t)__float2int_rn(fminf(255.0f, fmaxf(0.0f, v.g))),
                                  (uint8_t)__float2int_rn(fminf(255.0f, fmaxf(0.0f, v.b))));
            }
        }
    }
    out_v[out_idx] = hit;
    out_n[out_idx] = n;
    if (out_c) out_c[out_idx] = col;
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
        params_.truncation, params_.min_depth, params_.max_depth,
        fx, fy, cx, cy,
        R(0,0), R(0,1), R(0,2), t.x(),
        R(1,0), R(1,1), R(1,2), t.y(),
        R(2,0), R(2,1), R(2,2), t.z(),
        width, height, d_vertices, d_normals, d_colors
    );
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
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
        out_colors[out_idx] = make_uchar3(
            (uint8_t)__float2int_rn(fmaxf(0.0f, fminf(255.0f, v.r))),
            (uint8_t)__float2int_rn(fmaxf(0.0f, fminf(255.0f, v.g))),
            (uint8_t)__float2int_rn(fmaxf(0.0f, fminf(255.0f, v.b)))
        );
    }
}

void TSDFVolume::extractGlobalPointCloudGPU(std::vector<Eigen::Vector3f>& points_out,
                                            std::vector<uint8_t>&         colors_out) const
{
    int res = params_.resolution;
    int total_voxels = res * res * res;

    if (!d_pc_is_valid_) d_pc_is_valid_ = utils::make_cuda_unique<uint32_t>(total_voxels);
    if (!d_pc_offsets_) d_pc_offsets_ = utils::make_cuda_unique<uint32_t>(total_voxels);
    if (!d_pc_out_points_) d_pc_out_points_ = utils::make_cuda_unique<float3>(total_voxels);
    if (!d_pc_out_colors_) d_pc_out_colors_ = utils::make_cuda_unique<uchar3>(total_voxels);

    dim3 block(8, 8, 8);
    dim3 grid((res + block.x - 1) / block.x, (res + block.y - 1) / block.y, (res + block.z - 1) / block.z);

    classifyVoxelKernel<<<grid, block>>>(d_voxels_.get(), res, d_pc_is_valid_.get());
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Use Thrust for exclusive scan to get compaction offsets
    thrust::device_ptr<uint32_t> d_ptr_valid(d_pc_is_valid_.get());
    thrust::device_ptr<uint32_t> d_ptr_offsets(d_pc_offsets_.get());
    thrust::exclusive_scan(d_ptr_valid, d_ptr_valid + total_voxels, d_ptr_offsets);
    
    uint32_t total_points = thrust::reduce(d_ptr_valid, d_ptr_valid + total_voxels);

    if (total_points == 0) return;

    compactPointsKernel<<<grid, block>>>(
        d_voxels_.get(), d_pc_is_valid_.get(), d_pc_offsets_.get(),
        res, params_.voxel_size, 
        make_float3(params_.origin.x(), params_.origin.y(), params_.origin.z()),
        d_pc_out_points_.get(), d_pc_out_colors_.get()
    );
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    points_out.resize(total_points);
    colors_out.resize(total_points * 3);
    
    // Direct copy from device into the correctly sized Eigen/byte vectors
    CUDA_CHECK(cudaMemcpy(points_out.data(), d_pc_out_points_.get(), total_points * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colors_out.data(), d_pc_out_colors_.get(), total_points * sizeof(uchar3), cudaMemcpyDeviceToHost));
}

} // namespace tsdf
} // namespace kfusion

#endif // CUDA_ENABLED
