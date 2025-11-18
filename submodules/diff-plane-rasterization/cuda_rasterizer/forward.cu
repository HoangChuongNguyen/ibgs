/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



__device__ float3 bilinearInterpolate(int src_idx, const float* src_images, float2 projected_point, const int H, const int W, const int nb_channels){
	float warped_color[3] = {0};
	// Bilinear interpolate the color
	float u = projected_point.x;
	float v = projected_point.y;
	int u0 = (int)floorf(u);
	int v0 = (int)floorf(v);
	int u1 = u0 + 1;
	int v1 = v0 + 1;
	// Compute interpolation weights
	float wu = u - u0;
	float wv = v - v0;
	float wu1 = 1.0f - wu;
	float wv1 = 1.0f - wv;
	// Compute pixel indices in the flattened image
	int idx00 = v0 * W + u0;
	int idx01 = v0 * W + u1;
	int idx10 = v1 * W + u0;
	int idx11 = v1 * W + u1;
	for (int c = 0; c < nb_channels; c++) {  // Loop over RGB channels
		float I00 = src_images[src_idx * (nb_channels * H * W) + c * (H * W) + idx00]; // Top-left pixel
		float I01 = src_images[src_idx * (nb_channels * H * W) + c * (H * W) + idx01]; // Top-right pixel
		float I10 = src_images[src_idx * (nb_channels * H * W) + c * (H * W) + idx10]; // Bottom-left pixel
		float I11 = src_images[src_idx * (nb_channels * H * W) + c * (H * W) + idx11]; // Bottom-right pixel
		// Bilinear interpolation formula
		warped_color[c] = wu1 * wv1 * I00 + wu * wv1 * I01 + wu1 * wv * I10 + wu * wv * I11;
	}
	// If the image is grayscale or depth, replicate the value to all channels
	if (nb_channels == 1) {
		warped_color[1] = warped_color[0];
		warped_color[2] = warped_color[0];
	}
	float3 result = {warped_color[0], warped_color[1], warped_color[2]};
	return result;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const bool render_depth_only)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr && !render_depth_only)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}



// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// Optimized kernel (with explicit dot products)
template <uint32_t CHANNELS, uint32_t PLANE_PARAMS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float focal_x, const float focal_y,
    const float cx, const float cy,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ ref_to_src_list,
    const float* __restrict__ src_cam_pos,
    const float* __restrict__ src_images,
    cudaTextureObject_t texColor,
    const float* __restrict__ src_rendered_depths,
    cudaTextureObject_t texDepth,   
    const int nb_src_images,
    const int buffer_length,
    const float* __restrict__ cam_pos,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features,
    const float* __restrict__ all_map,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    float* __restrict__ buffer_cache_sum_median_weight,
    uint32_t* buffer_cache_low_median_contributor,
    uint32_t* buffer_cache_high_median_contributor,
    int* __restrict__ valid_src_indices,
    float* __restrict__ valid_src_sum_median_weight,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ out_normal_map,
    float* __restrict__ out_median_intersected_depth,
    float* __restrict__ out_cam_feat, 
    float* __restrict__ out_warped_image,
    float* __restrict__ out_min_depth_diff,
    float* __restrict__ out_camera_ray,
    int* __restrict__ out_use_first_src_frame_mask,
    const bool render_geo,
    const bool render_depth_only,
    const float depth_error_threshold)
{
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };
    const float2 ray = { (pixf.x - cx) / focal_x, (pixf.y - cy) / focal_y };
    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    // Precompute invariants
    const float inv_focal_x = 1.0f / focal_x;
    const float inv_focal_y = 1.0f / focal_y;
    const float pix_diff_x = pixf.x - cx;
    const float pix_diff_y = pixf.y - cy;
    const float epsilon = 1.0e-8f;

    float T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float C[CHANNELS] = { 0 };
    float normal_accum[NUM_NORMAL_CHANNELS] = { 0 };
    float median_intersected_depth = 0.0f;
    uint32_t low_median_contributor = 0;
    uint32_t high_median_contributor = 0;
    float warped_color_all[MAX_M * 3] = {0};
    float warped_depth_all[MAX_M] = {0};
    float buffer_intersected_depth[MAX_BUFFER_LENGTH] = {0};
    float buffer_weight[MAX_BUFFER_LENGTH] = {0};
    uint32_t buffer_median_contributor[MAX_BUFFER_LENGTH] = {0};
    const int before_cap = (buffer_length % 2 == 0) ? (buffer_length / 2) : ((buffer_length + 1) / 2);
    const int below_cap = buffer_length - before_cap;  // Simplified
    int before_ptr = 0;
    int below_count = 0;
    float total_buffer_weight = 0.0f;
    float weighted_depth_sum = 0.0f;

    // Shared memory for ref_to_src_list and src_cam_pos (load once per block)
    __shared__ float shared_ref_to_src[MAX_M * 16];
    __shared__ float shared_src_cam_pos[MAX_M * 3];
    if (threadIdx.x < nb_src_images) {
        for (int k = 0; k < 16; k++) {
            shared_ref_to_src[threadIdx.x * 16 + k] = ref_to_src_list[threadIdx.x * 16 + k];
        }
        for (int k = 0; k < 3; k++) {
            shared_src_cam_pos[threadIdx.x * 3 + k] = src_cam_pos[threadIdx.x * 3 + k];
        }
    }
    __syncthreads();

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        if (__syncthreads_count(done) == BLOCK_SIZE) break;

        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            contributor++;
            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) continue;

            float alpha = min(0.99f, con_o.w * __expf(power));  // Use __expf for speed
            if (alpha < 1.0f / 255.0f) continue;
            float test_T = T * (1.0f - alpha);
            if (test_T < 0.0001f) { done = true; continue; }
            const float aT = alpha * T;

            if (!render_depth_only) {
#pragma unroll
                for (int ch = 0; ch < CHANNELS; ch++) {
                    C[ch] += features[collected_id[j] * CHANNELS + ch] * aT;
                }
            }

            float intersected_depth = 0.0f;
            if (render_geo || render_depth_only) {
                intersected_depth = -all_map[collected_id[j] * PLANE_PARAMS + 4] /
                                    (all_map[collected_id[j] * PLANE_PARAMS + 0] * ray.x +
                                     all_map[collected_id[j] * PLANE_PARAMS + 1] * ray.y +
                                     all_map[collected_id[j] * PLANE_PARAMS + 2] + epsilon);
            }

            if (render_geo) {
#pragma unroll
                for (int ch = 0; ch < NUM_NORMAL_CHANNELS; ch++) {
                    normal_accum[ch] += all_map[collected_id[j] * PLANE_PARAMS + ch] * aT;
                }
                if (intersected_depth > 0.0f) {
                    if (T > 0.5f) {
                        buffer_intersected_depth[before_ptr] = intersected_depth;
                        buffer_weight[before_ptr] = aT;
                        buffer_median_contributor[before_ptr] = contributor;
                        before_ptr = (before_ptr + 1) % before_cap;
                    } else if (below_count < below_cap) {
                        int idx = before_cap + below_count;
                        buffer_intersected_depth[idx] = intersected_depth;
                        buffer_weight[idx] = aT;
                        buffer_median_contributor[idx] = contributor;
                        below_count++;
                    }
                }
            }

            if (render_depth_only && intersected_depth > 0.0f) {
                if (T > 0.5f) {
                    int idx = before_ptr;
                    total_buffer_weight -= buffer_weight[idx];
                    weighted_depth_sum -= buffer_weight[idx] * buffer_intersected_depth[idx];
                    buffer_intersected_depth[idx] = intersected_depth;
                    buffer_weight[idx] = aT;
                    before_ptr = (before_ptr + 1) % before_cap;
                    total_buffer_weight += aT;
                    weighted_depth_sum += aT * intersected_depth;
                } else if (below_count < below_cap) {
                    int idx = before_cap + below_count;
                    buffer_intersected_depth[idx] = intersected_depth;
                    buffer_weight[idx] = aT;
                    below_count++;
                    total_buffer_weight += aT;
                    weighted_depth_sum += aT * intersected_depth;
                }
				if (below_count== below_cap) {
					T = test_T;
					last_contributor = contributor;
					break; // Stop if buffer is full
				}
            }

            T = test_T;
            last_contributor = contributor;
        }
    }

    if (inside) {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;

        if (!render_depth_only) {
#pragma unroll
            for (int ch = 0; ch < CHANNELS; ch++) {
                out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
            }
        }

        if (render_depth_only) {
            median_intersected_depth = weighted_depth_sum / (total_buffer_weight + epsilon);
            out_median_intersected_depth[pix_id] = median_intersected_depth;
        }

		if (render_geo) {
			float total_buffer_weight_local = 0.0f;
			float total_buffer_weight_all_src[MAX_M] = {0};
			low_median_contributor = buffer_median_contributor[0];
			high_median_contributor = buffer_median_contributor[0];
		
			// Fused loop: Compute and cache transformed points for reuse
			float3 transformed_cache[MAX_M];  // Cache per src_idx (assume small MAX_M)
		#pragma unroll
			for (int i = 0; i < buffer_length; i++) {
				float weight = buffer_weight[i];
				if (weight != 0.0f) {
					float intersected_depth = buffer_intersected_depth[i];
					float3 intersected_point = {pix_diff_x * intersected_depth * inv_focal_x,
												pix_diff_y * intersected_depth * inv_focal_y,
												intersected_depth};
		
		#pragma unroll 4
					for (int src_idx = 0; src_idx < nb_src_images; src_idx++) {
						const float* ref_to_src = &shared_ref_to_src[src_idx * 16];
						float4 row0 = {ref_to_src[0], ref_to_src[1], ref_to_src[2], ref_to_src[3]};
						float4 row1 = {ref_to_src[4], ref_to_src[5], ref_to_src[6], ref_to_src[7]};
						float4 row2 = {ref_to_src[8], ref_to_src[9], ref_to_src[10], ref_to_src[11]};
						float4 point_homog = {intersected_point.x, intersected_point.y, intersected_point.z, 1.0f};
		
						float transformed_x = row0.x * point_homog.x + row0.y * point_homog.y + row0.z * point_homog.z + row0.w * point_homog.w;
						float transformed_y = row1.x * point_homog.x + row1.y * point_homog.y + row1.z * point_homog.z + row1.w * point_homog.w;
						float transformed_z = row2.x * point_homog.x + row2.y * point_homog.y + row2.z * point_homog.z + row2.w * point_homog.w;
						transformed_cache[src_idx] = {transformed_x, transformed_y, transformed_z};
		
						float inv_z = 1.0f / (transformed_z + epsilon);
						float2 projected_point = {transformed_x * focal_x * inv_z + cx,
												  transformed_y * focal_y * inv_z + cy};
		
						bool in_bounds = (projected_point.x >= 0.0f && projected_point.x <= (float)(W-1) &&
										  projected_point.y >= 0.0f && projected_point.y <= (float)(H-1));
		
						if (in_bounds) {
							float4 texC = tex2DLayered<float4>(texColor, projected_point.x + 0.5f, projected_point.y + 0.5f, src_idx);
							float3 warped_color = {texC.x, texC.y, texC.z};
							warped_color_all[src_idx * 3] += weight * warped_color.x;
							warped_color_all[src_idx * 3 + 1] += weight * warped_color.y;
							warped_color_all[src_idx * 3 + 2] += weight * warped_color.z;
							total_buffer_weight_all_src[src_idx] += weight;
						}
					}
					total_buffer_weight_local += weight;
					median_intersected_depth += weight * intersected_depth;
					low_median_contributor = min(low_median_contributor, buffer_median_contributor[i]);
					high_median_contributor = max(high_median_contributor, buffer_median_contributor[i]);
				}
			}

            buffer_cache_low_median_contributor[pix_id] = low_median_contributor;
            buffer_cache_high_median_contributor[pix_id] = high_median_contributor;
            buffer_cache_sum_median_weight[pix_id] = total_buffer_weight_local;
            median_intersected_depth /= (total_buffer_weight_local + epsilon);
            float3 median_intersected_point = {pix_diff_x * median_intersected_depth * inv_focal_x,
                                               pix_diff_y * median_intersected_depth * inv_focal_y,
                                               median_intersected_depth};

            // Ray computation (optimized with intrinsics)
            float3 translation = {viewmatrix[12], viewmatrix[13], viewmatrix[14]};
            float3 pc_world_trans = {median_intersected_point.x - translation.x,
                                     median_intersected_point.y - translation.y,
                                     median_intersected_point.z - translation.z};
            float3 median_intersected_point_world = {
                viewmatrix[0] * pc_world_trans.x + viewmatrix[1] * pc_world_trans.y + viewmatrix[2] * pc_world_trans.z,
                viewmatrix[4] * pc_world_trans.x + viewmatrix[5] * pc_world_trans.y + viewmatrix[6] * pc_world_trans.z,
                viewmatrix[8] * pc_world_trans.x + viewmatrix[9] * pc_world_trans.y + viewmatrix[10] * pc_world_trans.z
            };
            float3 ray_dir = {median_intersected_point_world.x - cam_pos[0],
                              median_intersected_point_world.y - cam_pos[1],
                              median_intersected_point_world.z - cam_pos[2]};
            float ray_len = sqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z) + epsilon;
            ray_dir.x /= ray_len;
            ray_dir.y /= ray_len;
            ray_dir.z /= ray_len;
            out_camera_ray[0 * H * W + pix_id] = ray_dir.x;
            out_camera_ray[1 * H * W + pix_id] = ray_dir.y;
            out_camera_ray[2 * H * W + pix_id] = ray_dir.z;

            int valid_src_count = 0;
            float min_depth_error = 1.0f;

#pragma unroll 4
            for (int src_idx = 0; src_idx < nb_src_images; src_idx++) {
                const float* ref_to_src = &shared_ref_to_src[src_idx * 16];
                float4 row0 = {ref_to_src[0], ref_to_src[1], ref_to_src[2], ref_to_src[3]};
                float4 row1 = {ref_to_src[4], ref_to_src[5], ref_to_src[6], ref_to_src[7]};
                float4 row2 = {ref_to_src[8], ref_to_src[9], ref_to_src[10], ref_to_src[11]};
                float4 point_homog = {median_intersected_point.x, median_intersected_point.y, median_intersected_point.z, 1.0f};

                // Explicit dot product computations
                float transformed_x = row0.x * point_homog.x + row0.y * point_homog.y + row0.z * point_homog.z + row0.w * point_homog.w;
                float transformed_y = row1.x * point_homog.x + row1.y * point_homog.y + row1.z * point_homog.z + row1.w * point_homog.w;
                float transformed_z = row2.x * point_homog.x + row2.y * point_homog.y + row2.z * point_homog.z + row2.w * point_homog.w;
                float3 transformed_point = {transformed_x, transformed_y, transformed_z};

                float inv_z = 1.0f / (transformed_point.z + epsilon);
                float2 projected_point = {transformed_point.x * focal_x * inv_z + cx,
                                          transformed_point.y * focal_y * inv_z + cy};

                bool in_bounds = (projected_point.x >= 0.0f && projected_point.x <= (float)(W-1) &&
                                  projected_point.y >= 0.0f && projected_point.y <= (float)(H-1));

                float warped_depth = 0.0f;
                if (in_bounds) {
                    warped_depth = tex2DLayered<float>(texDepth, projected_point.x + 0.5f, projected_point.y + 0.5f, src_idx);
                }
                warped_depth_all[src_idx] = warped_depth;

                float depth_error = fabsf(warped_depth - transformed_point.z) * inv_z;

                if (warped_depth > 0.0f && depth_error < depth_error_threshold) {
                    float inv_weight = 1.0f / (total_buffer_weight_all_src[src_idx] + epsilon);
#pragma unroll
                    for (int c = 0; c < 3; c++) {
                        warped_color_all[src_idx * 3 + c] *= inv_weight;
                        out_cam_feat[valid_src_count * 4 * H * W + c * H * W + pix_id] = cam_pos[c] - shared_src_cam_pos[src_idx * 3 + c];
                        out_warped_image[valid_src_count * 3 * H * W + c * H * W + pix_id] = warped_color_all[src_idx * 3 + c];
                    }

                    float3 source_dir = {median_intersected_point_world.x - shared_src_cam_pos[src_idx * 3],
                                         median_intersected_point_world.y - shared_src_cam_pos[src_idx * 3 + 1],
                                         median_intersected_point_world.z - shared_src_cam_pos[src_idx * 3 + 2]};
                    float source_len = sqrtf(source_dir.x * source_dir.x + source_dir.y * source_dir.y + source_dir.z * source_dir.z) + epsilon;
                    source_dir.x /= source_len;
                    source_dir.y /= source_len;
                    source_dir.z /= source_len;

                    // Explicit dot product for ray_dir_diff
                    float ray_dir_diff = source_dir.x * ray_dir.x + source_dir.y * ray_dir.y + source_dir.z * ray_dir.z;
                    out_cam_feat[valid_src_count * 4 * H * W + 3 * H * W + pix_id] = ray_dir_diff;

                    if (src_idx == 0) out_use_first_src_frame_mask[pix_id] = 1;
                    valid_src_indices[valid_src_count * H * W + pix_id] = src_idx;
                    valid_src_sum_median_weight[valid_src_count * H * W + pix_id] = total_buffer_weight_all_src[src_idx];
                    valid_src_count++;
                    min_depth_error = min(min_depth_error, depth_error);
                    if (valid_src_count == M) break;  // Assume M is defined as MAX_M
                }
            }
            if (valid_src_count <= M - 1) valid_src_indices[valid_src_count * H * W + pix_id] = -1;
            out_min_depth_diff[pix_id] = min_depth_error;

            out_median_intersected_depth[pix_id] = median_intersected_depth;
#pragma unroll
            for (int ch = 0; ch < NUM_NORMAL_CHANNELS; ch++) {
                out_normal_map[ch * H * W + pix_id] = normal_accum[ch];
            }
        }
    }
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const float cx, const float cy,
	const float* viewmatrix,
	// My change here
	const float* ref_to_src_list,
	const float* src_cam_pos,
	const float* src_images,
	cudaTextureObject_t texColor,
	// const float* src_rendered_images,
	const float* src_rendered_depths,
	cudaTextureObject_t texDepth,
	const int nb_src_images,
	const int buffer_length,
	const float depth_error_threshold,
	// End my change here
	const float* cam_pos,
	const float2* means2D,
	const float* colors,
	const float* all_map,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	// My change here
	// float* buffer_cache_intersected_depth,
	float* buffer_cache_sum_median_weight,
	// float* buffer_cache_weighted_sum_intersected_depth,
	// float* buffer_cache_weighted_sum_warped_depth_all,
	// float* buffer_cache_weighted_sum_warped_color_all,
	// float* buffer_cache_weighted_sum_median_weight_all_src, 
	uint32_t* buffer_cache_low_median_contributor,
	uint32_t* buffer_cache_high_median_contributor,
	int* valid_src_indices,
	float* valid_src_sum_median_weight,
	// End my change here
	const float* bg_color,
	float* out_color,
	// int* out_observe,
	float* out_normal_map,
	// My change here
	float* out_median_intersected_depth,
	// float* out_median_warped_depth,
	// float* out_median_warped_color,
	// Color aggregation feature
	float* out_cam_feat, 
	float* out_warped_image,
	// float* out_color_aggregation_warped_rendered_image,
	float* out_min_depth_diff,
	float* out_camera_ray,
	// --- Extra params for exposure ---
	int* out_use_first_src_frame_mask,
	// End change here
	const bool render_geo,
	const bool render_depth_only)
{
	renderCUDA<NUM_CHANNELS,NUM_PLANE_PARAMS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		cx, cy,
		viewmatrix,
		ref_to_src_list,
		src_cam_pos,
		src_images,
		texColor, 
		src_rendered_depths, 
		texDepth,
		nb_src_images,
		buffer_length,
		cam_pos,
		means2D,
		colors,
		all_map,
		conic_opacity,
		final_T,
		n_contrib,
		buffer_cache_sum_median_weight,
		buffer_cache_low_median_contributor,
		buffer_cache_high_median_contributor,
		valid_src_indices,
		valid_src_sum_median_weight,
		bg_color,
		out_color,
		out_normal_map,
		out_median_intersected_depth,
		out_cam_feat, 
		out_warped_image,
		out_min_depth_diff,
		out_camera_ray,
		out_use_first_src_frame_mask,
		// End chnge here
		render_geo,
		render_depth_only,
		depth_error_threshold);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const bool render_depth_only)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		render_depth_only
		);
}
