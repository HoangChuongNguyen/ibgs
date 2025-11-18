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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
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
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		const bool render_depth_only);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float focal_x, const float focal_y,
		const float cx, const float cy,
		const float* viewmatrix,
		const float* ref_to_src_list,
		const float* src_cam_pos,
		const float* src_images,
		cudaTextureObject_t texColor,
		const float* src_rendered_depths,
		cudaTextureObject_t texDepth,
	const int nb_src_images,
	const int buffer_length,
	const float depth_error_threshold,
	const float* cam_pos,
		const float2* points_xy_image,
		const float* features,
		const float* all_map,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		float* buffer_cache_sum_median_weight,
		uint32_t* buffer_cache_low_median_contributor,
		uint32_t* buffer_cache_high_median_contributor,
		int* valid_src_indices,
		float* valid_src_sum_median_weight,
		const float* bg_color,
		float* out_color,
		float* out_normal_map,
		float* out_median_intersected_depth,
		float* out_cam_feat, 
		float* out_warped_image,
		float* out_min_depth_diff,
		float* out_camera_ray,
		int* out_use_first_src_frame_mask,
		const bool render_geo,
		const bool render_depth_only);
}


#endif
