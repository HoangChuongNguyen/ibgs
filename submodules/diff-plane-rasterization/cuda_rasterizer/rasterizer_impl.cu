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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Packs src_images (M × 3 × H × W) into a float4 array (M × H × W), alpha=1.0
__global__ void packRGBA(
    const float* __restrict__ src,  // [ layer0 R… G… B… | layer1 … ]
    float4*      __restrict__ dst,  // [ (R,G,B,1) … ]
    int          W,
    int          H,
    int          M)
{
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    size_t total = size_t(W)*H*M;
    if (idx >= total) return;
    // which layer and which pixel
    int layer = idx / (W*H);
    int pix   = idx % (W*H);
    int y     = pix / W;
    int x     = pix % W;
    // compute offsets in the source float array
    size_t base = size_t(layer) * 3 * W * H;
    size_t ofs  = y*W + x;
    float r = src[base +       ofs];
    float g = src[base + W*H + ofs];
    float b = src[base + 2*W*H + ofs];
    dst[idx]  = make_float4(r, g, b, 1.0f);
}

struct LayeredTextures
{
	cudaTextureObject_t color = 0;
	cudaTextureObject_t depth = 0;
	cudaArray_t color_array = nullptr;
	cudaArray_t depth_array = nullptr;
	float4* packed_rgba = nullptr;
};

static LayeredTextures createLayeredTextures(
	int width,
	int height,
	int nb_layers,
	const float* src_images,
	const float* src_rendered_depths)
{
	LayeredTextures textures;
	if (nb_layers <= 0)
		return textures;

	cudaChannelFormatDesc chanDescC = cudaCreateChannelDesc<float4>();
	cudaChannelFormatDesc chanDescD = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&textures.color_array, &chanDescC,
		make_cudaExtent(width, height, nb_layers),
		cudaArrayLayered);
	cudaMalloc3DArray(&textures.depth_array, &chanDescD,
		make_cudaExtent(width, height, nb_layers),
		cudaArrayLayered);

	cudaMemcpy3DParms copyParams = {};

	size_t totalPixels = size_t(width) * height * nb_layers;
	cudaMalloc(reinterpret_cast<void**>(&textures.packed_rgba), totalPixels * sizeof(float4));

	int threads = 256;
	int blocks = int((totalPixels + threads - 1) / threads);
	packRGBA<<<blocks, threads>>>(
		src_images,
		textures.packed_rgba,
		width, height,
		nb_layers);
	cudaDeviceSynchronize();

	copyParams.srcPtr = make_cudaPitchedPtr(
		textures.packed_rgba,
		width * sizeof(float4),
		width, height);
	copyParams.dstArray = textures.color_array;
	copyParams.extent = make_cudaExtent(width, height, nb_layers);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);

	copyParams.srcPtr = make_cudaPitchedPtr(
		reinterpret_cast<void*>(const_cast<float*>(src_rendered_depths)),
		width * sizeof(float),
		width, height);
	copyParams.dstArray = textures.depth_array;
	cudaMemcpy3D(&copyParams);

	cudaResourceDesc resDescC = {};
	resDescC.resType = cudaResourceTypeArray;
	resDescC.res.array.array = textures.color_array;
	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaCreateTextureObject(&textures.color, &resDescC, &texDesc, nullptr);

	cudaResourceDesc resDescD = resDescC;
	resDescD.res.array.array = textures.depth_array;
	cudaCreateTextureObject(&textures.depth, &resDescD, &texDesc, nullptr);

	return textures;
}

static void destroyLayeredTextures(LayeredTextures& textures)
{
	if (textures.packed_rgba)
		cudaFree(textures.packed_rgba);
	if (textures.color)
		cudaDestroyTextureObject(textures.color);
	if (textures.depth)
		cudaDestroyTextureObject(textures.depth);
	if (textures.color_array)
		cudaFreeArray(textures.color_array);
	if (textures.depth_array)
		cudaFreeArray(textures.depth_array);
	textures = {};
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.buffer_cache_sum_median_weight, N, 128);
	obtain(chunk, img.buffer_cache_low_median_contributor, N, 128);
	obtain(chunk, img.buffer_cache_high_median_contributor, N, 128);
	obtain(chunk, img.valid_src_indices, N*M, 128);
	obtain(chunk, img.valid_src_sum_median_weight, N*M, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	const float*  ref_to_src_list,
	const float* src_cam_pos,
	const float*  src_images,
	const float*  src_rendered_depths,
	const int nb_src_images,
	const int buffer_length,
	const float depth_error_threshold,
	// End change hre
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	float* out_normal_map,
	float* out_median_intersected_depth,
	float* out_cam_feat, 
	float* out_warped_image,
	float* out_min_depth_diff,
	float* out_camera_ray,
	int* out_use_first_src_frame_mask,
	const bool render_geo,
	const bool render_depth_only,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// ─── HOST SETUP FOR LAYERED TEXTURES ───────────────────────────────────
	LayeredTextures layeredTextures = createLayeredTextures(
		width, height, nb_src_images,
		src_images,
		src_rendered_depths);
	// ────────────────────────────────────────────────────────────────────────


	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		render_depth_only
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		float(width*0.5f), float(height*0.5f),
		viewmatrix,
		ref_to_src_list,
		src_cam_pos,
		src_images,
		layeredTextures.color,
		src_rendered_depths, 
		layeredTextures.depth,
		nb_src_images, 
		buffer_length,
		depth_error_threshold,
		cam_pos,
		geomState.means2D,
		feature_ptr,
		all_map,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.buffer_cache_sum_median_weight,
		imgState.buffer_cache_low_median_contributor,
		imgState.buffer_cache_high_median_contributor,
		imgState.valid_src_indices,
		imgState.valid_src_sum_median_weight,
		background,
		out_color,
		out_normal_map,
		out_median_intersected_depth,
		out_cam_feat, 
		out_warped_image,
		out_min_depth_diff,
		out_camera_ray,
		out_use_first_src_frame_mask,
		render_geo,
		render_depth_only), debug)
	// ─── CLEANUP ───────────────────────────────────────────────────────────
	destroyLayeredTextures(layeredTextures);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
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
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// ─── HOST SETUP FOR LAYERED TEXTURES ───────────────────────────────────
	LayeredTextures layeredTextures = createLayeredTextures(
		width, height, nb_src_images,
		src_images,
		src_rendered_depths);
	// ────────────────────────────────────────────────────────────────────────


	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		ref_to_src_list,
		src_cam_pos,
		src_images,
		layeredTextures.color,
		src_rendered_depths,
		layeredTextures.depth,
		nb_src_images, 
		color_ptr,
		all_maps,
        normal_map_pixels,
		intersected_depth_pixels, 
		warped_image_pixels, 
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.buffer_cache_sum_median_weight,
		imgState.buffer_cache_low_median_contributor,
		imgState.buffer_cache_high_median_contributor,
		imgState.valid_src_indices,
		imgState.valid_src_sum_median_weight,
		dL_dpix,
		dL_dout_normal_map,
		dL_dout_median_intersected_depth,
		dL_dout_warped_image, 
		(float3*)dL_dmean2D,
		(float3*)dL_dmean2D_abs,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dall_map,
		render_geo), debug)

	// ─── CLEANUP ───────────────────────────────────────────────────────────
	destroyLayeredTextures(layeredTextures);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
