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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* all_map,
			const float* viewmatrix,
			const float* projmatrix,
			// My change here
			const float*  ref_to_src_list,
			const float* src_cam_pos,
			const float*  src_images,
		const float*  src_rendered_depths,
		const int nb_src_images,
		const int buffer_length,
		const float depth_error_threshold,
		// End change here
		const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii,
			float* out_normal_map,
			// My change here
			float* out_median_intersected_depth,
			// Color aggregation feature
			float* out_cam_feat, 
			float* out_warped_image,
			float* out_min_depth_diff,
			float* out_camera_ray,
			// --- Extra params for exposure ---
			int* out_use_first_src_frame_mask,
			// End change here
			const bool render_geo,
			const bool render_depth_only,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const float* normal_map_pixels,
			const float* intersected_depth_pixels,
			const float* warped_image_pixels,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* all_maps,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			// My change here
			const float* ref_to_src_list,
			const float* src_cam_pos,
			const float* src_images,
		const float* src_rendered_depths,
		const int nb_src_images, 
		// End change here
		const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dout_normal_map,
			// My change here
			const float* dL_dout_median_intersected_depth,
			const float* dL_dout_warped_image,
			// End change here
			float* dL_dmean2D,
			float* dL_dmean2D_abs,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dall_map,
			const bool render_geo,
			bool debug);
	};
};

#endif
