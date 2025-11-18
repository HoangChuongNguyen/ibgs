
import torch
import math
from typing import Optional
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
from scene import Scene
from utils.general_utils import generate_image_coordinates
import random
import numpy as np

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def depth_transform_src2ref(ref_cam, src_cam):
    ref_image = ref_cam.original_image.float().cuda() # 3 H W
    ref_to_world = ref_cam.world_view_transform.T.inverse()
    src_to_world = src_cam.world_view_transform.T.inverse()
    src_to_ref = torch.inverse(ref_to_world) @ src_to_world # 4 4
    K_ref = ref_cam.get_k() # 3 3
    _, height, width = src_cam.rendered_depth.shape
    grid = generate_image_coordinates(height, width) # 3 H W
    grid[[1,0]] = grid[[0,1]]
    xyz = torch.inverse(K_ref) @ (grid*src_cam.rendered_depth.float().cuda()).view(3,-1) # 3 H*W
    xyz = src_to_ref[:3,:3] @ xyz + src_to_ref[:3,[-1]] # 3 H*W
    return xyz[[-1]].view(1,height,width)

def render_depth(viewpoint_camera, pc : GaussianModel, scene: Scene, pipe, args, bg_color : torch.Tensor, 
            learnt_normal: bool, nb_src_frames: int, buffer_length:int, depth_error_threshold: Optional[float] = None,
            scaling_modifier = 1.0, override_color = None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if depth_error_threshold is None:
        depth_error_threshold = getattr(args, "depth_error_threshold", 0.01)
    depth_error_threshold = float(depth_error_threshold)

    nb_src_frames = 1
    ref_to_src_list = torch.zeros((nb_src_frames, 4*4), device="cuda")
    src_images = torch.zeros((nb_src_frames, 3, viewpoint_camera.image_height*viewpoint_camera.image_width), device="cuda")
    src_rendered_depths = torch.zeros((nb_src_frames, 1, viewpoint_camera.image_height*viewpoint_camera.image_width), device="cuda")
    src_cam_pos = torch.zeros((nb_src_frames, 3), device="cuda")

    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            ref_to_src_list=ref_to_src_list,
            src_cam_pos=src_cam_pos, 
            src_images=src_images,
            src_rendered_depths=src_rendered_depths,
            nb_src_images=nb_src_frames,
            buffer_length=buffer_length,
            depth_error_threshold=depth_error_threshold,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=False,
            render_depth_only=True,
            debug=pipe.debug
        )
    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)
    if learnt_normal: global_normal, offset_global = pc.get_normal(viewpoint_camera)
    else: global_normal = pc.get_normal_w_smallest_axis(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    global_distance = -(global_normal * means3D).sum(-1) 
    if learnt_normal: global_distance += offset_global.squeeze()
    local_distance = (global_distance - torch.sum(local_normal*viewpoint_camera.world_view_transform[[3],:3], dim=1))
    local_distance = local_distance.abs()
    input_all_map = torch.zeros((means3D.shape[0], 5), device='cuda').float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance
	
    (_, _, _,  out_median_intersected_depth, _, _, _, _, _) = rasterizer(means3D = means3D,
                                                            means2D = means2D,
                                                            means2D_abs = means2D_abs,
                                                            shs = shs,
                                                            colors_precomp = colors_precomp,
                                                            opacities = opacity,
                                                            scales = scales,
                                                            rotations = rotations,
                                                            all_map = input_all_map,
                                                            cov3D_precomp = cov3D_precomp)
    return out_median_intersected_depth
    

def render(viewpoint_camera, pc : GaussianModel, scene: Scene, pipe, args, bg_color : torch.Tensor, 
            learnt_normal: bool, nb_src_frames: int, buffer_length:int, depth_error_threshold: Optional[float] = None,
            scaling_modifier = 1.0, override_color = None, 
            app_model: AppModel=None, render_geo = True, return_depth_normal = True, 
            do_find_closest_frame=False, do_render_src_depth=False, render_depth_only=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    if depth_error_threshold is None:
        depth_error_threshold = getattr(args, "depth_error_threshold", 0.01)
    depth_error_threshold = float(depth_error_threshold)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color


    if render_geo:
        # Find neighboring source frames based on camera pose. 
        if do_find_closest_frame:
            world_view_transforms = viewpoint_camera.world_view_transform.T
            camera_center = viewpoint_camera.camera_center # 3 
            R = torch.tensor(viewpoint_camera.R).float()
            center_ray = torch.tensor([0.0, 0.0, 1.0], device="cuda").float()
            center_ray = center_ray @ R.transpose(-1, -2)
            diss_test = torch.norm(camera_center.unsqueeze(0) - scene.camera_centers, dim=-1) # M 3
            diss_test = diss_test.detach().cpu().numpy()
            tmp = torch.sum(center_ray.unsqueeze(0)*scene.center_rays, dim=-1) # M 1
            angles = torch.arccos(tmp)*180/torch.pi
            angles = angles.detach().cpu().numpy()
            sorted_indices = np.lexsort((angles, diss_test)) # sort by angle, then by distance
            mask = (angles[sorted_indices] < args.multi_view_max_angle) & \
                (diss_test[sorted_indices] > args.multi_view_min_dis) & \
                (diss_test[sorted_indices] < args.multi_view_max_dis)
            sorted_indices = sorted_indices[mask]
            multi_view_num = min(args.multi_view_num, len(sorted_indices))
            sorted_indices = sorted_indices[:multi_view_num]
            sorted_indices = sorted_indices.tolist()
            if args.enable_exposure_correction:
                relative_poses = torch.matmul(world_view_transforms.unsqueeze(0), torch.inverse(scene.world_view_transforms)) # nb_ref x 4 x 4
                cam_diff_list = torch.mean(torch.abs(relative_poses-torch.eye(4).float().cuda().unsqueeze(0)), dim=[1,2]).detach().cpu().numpy()
                test_cam_diff = cam_diff_list[sorted_indices]
                smallest_cam_diff_indice = sorted_indices[np.argmin(test_cam_diff)]
                sorted_indices.remove(smallest_cam_diff_indice)
                sorted_indices = [smallest_cam_diff_indice] + sorted_indices
            sorted_indices = np.array(sorted_indices)
            nearest_frame_list = sorted_indices
        else: nearest_frame_list = viewpoint_camera.nearest_id

        if len(nearest_frame_list) == 0: 
            nb_src_frames = 1
            ref_to_src_list = torch.zeros((nb_src_frames, 4*4), device="cuda")
            src_images = torch.zeros((nb_src_frames, 3, viewpoint_camera.image_height*viewpoint_camera.image_width), device="cuda")
            src_rendered_depths = torch.zeros((nb_src_frames, 1, viewpoint_camera.image_height*viewpoint_camera.image_width), device="cuda")
            src_cam_pos = torch.zeros((nb_src_frames, 3), device="cuda")
        else:
            nb_src_frames = min(nb_src_frames, len(nearest_frame_list))
            if args.shuffle_source_frame:
                selected_src_indices = random.sample(nearest_frame_list, nb_src_frames)
            else:
                selected_src_indices = nearest_frame_list[:nb_src_frames]

            src_images = scene.original_image_list[selected_src_indices]
            # Get depth of nearby camera
            if do_render_src_depth:
                src_rendered_depths = []
                for src_idx in selected_src_indices:
                    src_view = scene.getTrainCameras()[src_idx]
                    src_rendered_depth = render_depth(src_view, pc, scene, pipe, args, bg_color, 
                                                    learnt_normal, nb_src_frames, buffer_length, depth_error_threshold,
                                                    scaling_modifier, override_color)
                    src_rendered_depths.append(src_rendered_depth)
                src_rendered_depths = torch.stack(src_rendered_depths, dim=0)
            else:
                src_rendered_depths = scene.rendered_depth_list[selected_src_indices] # M 1 H W

            # Post process stuffs
            world_to_src = scene.world_view_transforms[selected_src_indices]
            src_to_world = torch.inverse(world_to_src)
            ref_to_world = viewpoint_camera.world_view_transform.T.inverse()
            ref_to_src_list = world_to_src @ ref_to_world.unsqueeze(0) # 4 4 
            src_cam_pos = src_to_world[:, :3, 3] 

            ref_to_src_list = world_to_src @ ref_to_world.unsqueeze(0).cuda() # 4 4 
            src_cam_pos = src_to_world[:, :3, 3].cuda() 
            src_rendered_depths = src_rendered_depths.cuda()
            src_images = src_images.cuda()

    else:
        nb_src_frames = 1
        ref_to_src_list = torch.zeros((nb_src_frames, 4*4), device="cuda")
        src_images = torch.zeros((nb_src_frames, 3, viewpoint_camera.image_height*viewpoint_camera.image_width), device="cuda")
        src_rendered_depths = torch.zeros((nb_src_frames, 1, viewpoint_camera.image_height*viewpoint_camera.image_width), device="cuda")
        src_cam_pos = torch.zeros((nb_src_frames, 3), device="cuda")

    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            ref_to_src_list=ref_to_src_list,
            src_cam_pos=src_cam_pos, 
            src_images=src_images,
            src_rendered_depths=src_rendered_depths,
            nb_src_images=nb_src_frames,
            buffer_length=buffer_length,
            depth_error_threshold=depth_error_threshold,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=render_geo,
            render_depth_only=render_depth_only,
            debug=pipe.debug
        )


    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if render_geo or render_depth_only:
        if learnt_normal: global_normal, offset_global = pc.get_normal(viewpoint_camera)
        else: global_normal = pc.get_normal_w_smallest_axis(viewpoint_camera)
        local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
        global_distance = -(global_normal * means3D).sum(-1) 
        if learnt_normal: global_distance += offset_global.squeeze()
        local_distance = (global_distance - torch.sum(local_normal*viewpoint_camera.world_view_transform[[3],:3], dim=1))
        local_distance = local_distance.abs()
        input_all_map = torch.zeros((means3D.shape[0], 5), device='cuda').float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance
    else: input_all_map = None
	
    (rendered_image, radii, 
        out_normal_map, 
        out_median_intersected_depth, 
            out_cam_feat, out_warped_image,
            out_min_depth_diff, out_camera_ray, use_first_src_frame_mask) = rasterizer(means3D = means3D,
                                                                                        means2D = means2D,
                                                                                        means2D_abs = means2D_abs,
                                                                                        shs = shs,
                                                                                        colors_precomp = colors_precomp,
                                                                                        opacities = opacity,
                                                                                        scales = scales,
                                                                                        rotations = rotations,
                                                                                        all_map = input_all_map,
                                                                                        cov3D_precomp = cov3D_precomp)

    if render_geo:
        rendered_normal = out_normal_map[0:3]
    else:
        rendered_normal = None
    
    if return_depth_normal:
        median_intersected_depth_normal = render_normal(viewpoint_camera, out_median_intersected_depth.squeeze()) 
        median_intersected_depth_normal_norm = torch.norm(median_intersected_depth_normal, dim=0, keepdim=True)
        median_intersected_depth_normal = median_intersected_depth_normal / (median_intersected_depth_normal_norm+1e-8)
    else: median_intersected_depth_normal = None

    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid, device='cuda')]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
    else: app_image = None

    return_dict =  {"render": rendered_image,
                    "app_image": app_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "rendered_normal": rendered_normal,
                    "median_intersected_depth": out_median_intersected_depth,
                    "median_intersected_depth_normal": median_intersected_depth_normal,
                    "cam_feat": out_cam_feat,
                    "warped_image": out_warped_image,
                    "min_depth_diff": out_min_depth_diff,
                    "camera_ray": out_camera_ray,
                    "use_first_src_frame_mask": use_first_src_frame_mask
                    }

    return return_dict
