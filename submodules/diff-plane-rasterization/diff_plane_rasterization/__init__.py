#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    means2D_abs,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    all_map,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_map,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_maps,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            all_maps,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            # My change here
            raster_settings.ref_to_src_list,
            raster_settings.src_cam_pos,
            raster_settings.src_images,
            raster_settings.src_rendered_depths,
            raster_settings.nb_src_images,
        raster_settings.buffer_length, 
        raster_settings.depth_error_threshold,
            # End my change here
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.render_geo,
            raster_settings.render_depth_only,  
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                (num_rendered, color, radii, 
                    out_normal_map, 
                    out_median_intersected_depth, 
                    out_cam_feat, out_warped_image,
                    out_min_depth_diff,
                    out_camera_ray, out_use_first_src_frame,
                    geomBuffer, binningBuffer, imgBuffer) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            (num_rendered, color, radii, 
                # out_observe, 
                out_normal_map, 
                # My change here
                out_median_intersected_depth, 
                # out_median_warped_depth, out_median_warped_color,	
                # Color aggregation feature
                out_cam_feat, out_warped_image,
                # out_color_aggregation_warped_rendered_image, 
                out_min_depth_diff,
                out_camera_ray, out_use_first_src_frame,
                # End my change here 
                geomBuffer, binningBuffer, imgBuffer) = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(out_normal_map, 
                            # My change here
                            out_median_intersected_depth, 
                            # out_median_warped_depth, out_median_warped_color,	
                            # Color aggregation feature
                            out_warped_image,
                            # End my change here 
                            colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return (color, radii, 
                # out_observe, 
                out_normal_map, 
                # My change here
                out_median_intersected_depth, 
                # out_median_warped_depth, out_median_warped_color, 
                # Color aggregation feature
                out_cam_feat, out_warped_image,
                # out_color_aggregation_warped_rendered_image, 
                out_min_depth_diff,
                out_camera_ray, out_use_first_src_frame
                # End my change here 
                )

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, 
                #  grad_out_observe, 
                 grad_out_normal_map, 
                # My change here
                grad_out_median_intersected_depth, 
                # grad_out_median_warped_depth, grad_out_median_warped_color,
                # Color aggregation feature
                grad_out_cam_feat, grad_out_warped_image,
                # grad_out_color_aggregation_warped_rendered_image, 
                grad_out_min_depth_diff,
                grad_out_camera_ray, grad_out_use_first_src_frame
                # End my change here 
                ):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (normal_map_pixels, 
            # My change here
            median_intersected_depth_pixels, 
            # median_warped_depth_pixels, median_warped_color_pixels,
            warped_image_pixels,
            # End my change here
            colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                normal_map_pixels,
                median_intersected_depth_pixels, 
                warped_image_pixels, 
                means3D, 
                radii, 
                colors_precomp, 
                all_maps,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                # My change here
                raster_settings.ref_to_src_list,
                raster_settings.src_cam_pos,
                raster_settings.src_images,
                raster_settings.src_rendered_depths,
                raster_settings.nb_src_images,
                # End my change here
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_normal_map,
                # My change here
                grad_out_median_intersected_depth, 
                # grad_out_median_warped_depth, 
                # grad_out_median_warped_color,
                grad_out_warped_image, 
                # End my change here
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.render_geo,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_all_map = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_all_map = _C.rasterize_gaussians_backward(*args)
        # print(f"grad_means2D {grad_means2D.sum()}, grad_means2D_abs {grad_means2D_abs.sum()}")

        grads = (
            grad_means3D,
            grad_means2D,
            grad_means2D_abs,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_all_map,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    # My change here
    ref_to_src_list : torch.Tensor
    src_cam_pos: torch.Tensor
    src_images : torch.Tensor
    # src_rendered_images : torch.Tensor
    src_rendered_depths : torch.Tensor
    nb_src_images: int
    buffer_length : int
    depth_error_threshold : float
    # End my change here
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    render_geo : bool
    render_depth_only : bool  # My change here
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, means2D_abs, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, all_map=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if all_map is None:
            all_map = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_abs,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            all_map,
            raster_settings, 
        )
