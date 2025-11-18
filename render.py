
import os
import sys
import torch
from scene import Scene
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import colorize
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import trimesh, copy
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
from color_aggregation_network import ColorFusionResidualNet, fuse_color
import time
from matplotlib import pyplot as plt
import pytorch3d
import random
from torch.cuda.amp import autocast
from contextlib import nullcontext

torch.backends.cudnn.benchmark = True

DEFAULT_DEPTH_TYPE = "median_intersected_depth"

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0


def get_folder_file_size(path):
    """
    Return the disk usage of the given file or directory in megabytes (MB),
    matching `du -m`, where each block is 512 bytes and 1 MB = 1024 * 1024 bytes.
    """
    total_blocks = 0

    def blocks(p: str) -> int:
        try:
            # st_blocks = number of 512-byte blocks actually allocated
            return os.lstat(p).st_blocks
        except OSError:
            return 0
    # If it's a directory, include the dir inode itself plus all contained files
    if os.path.isdir(path) and not os.path.islink(path):
        for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
            total_blocks += blocks(dirpath)
            for fname in filenames:
                total_blocks += blocks(os.path.join(dirpath, fname))
    else:
        # file or symlink
        total_blocks += blocks(path)

    total_bytes = total_blocks * 512
    return total_bytes / (1024 ** 2)


def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, args, opt, background, 
               color_aggregation_network,
               learnt_normal, nb_src_frames, buffer_length,
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False):
    # box
    js_file = f"{scene.source_path}/transforms.json"
    bounds = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            if "aabb_range" in meta:
                bounds = (np.array(meta["aabb_range"]))
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    if opt.use_color_aggregation: 
        render_aggregate_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_aggregate")
        residual_path = os.path.join(model_path, name, "ours_{}".format(iteration), "predicted_residual")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    if opt.use_color_aggregation: 
        makedirs(render_aggregate_path, exist_ok=True)
        makedirs(residual_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)

    depth_error_threshold = opt.depth_error_threshold
    rendered_time_list = []
    if name=="test":
        # Test the FPS
        for i in range(6):
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                torch.cuda.synchronize()
                time_start = time.time()
                out = render(view, gaussians, scene, pipeline, args, background, 
                            learnt_normal=learnt_normal, nb_src_frames=nb_src_frames, buffer_length=buffer_length, depth_error_threshold=depth_error_threshold,
                            app_model=app_model, do_find_closest_frame=True, do_render_src_depth=True, render_geo=True, return_depth_normal=volume is not None)
                aggregate_image = predicted_residual = None
                if opt.use_color_aggregation and color_aggregation_network is not None:
                    autocast_ctx = autocast() if opt.enable_mix_precision else nullcontext()
                    with autocast_ctx:
                        fusion_out_dict = fuse_color(
                            out,
                            color_aggregation_network=color_aggregation_network,
                            iter_count=None,
                            burn_start=None,
                            burn_end=None,
                            iteration=iteration,
                            opts=opt,
                        )
                    if fusion_out_dict is not None:
                        aggregate_image = fusion_out_dict["image_pred"]
                        predicted_residual = fusion_out_dict["residual"]
                torch.cuda.synchronize()
                time_end = time.time()

                if i!=0: rendered_time_list.append((time_end - time_start)*1000)
        FPS = 1000/(np.mean(rendered_time_list))
        print("FPS: ", FPS)

        # Get the number of Gaussians
        num_gaussians = len(gaussians.get_xyz)

        # To mimic the real testing environment, we store src images then reload them
        misc_path = f'{model_path}/test_time_data/ours_{iteration}'
        makedirs(f"{misc_path}/images", exist_ok=True)
        intrinsic_list = []
        extrinsic_list = []
        for idx, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering progress")):
            # Save images
            gt = view.original_image
            torchvision.utils.save_image(gt, f"{misc_path}/images/{view.image_name}.{args.src_image_ext}")
            gt_jpg  = plt.imread(f"{misc_path}/images/{view.image_name}.{args.src_image_ext}")
            if np.max(gt_jpg) > 1.0: gt_jpg = gt_jpg/255.0
            scene.original_image_list[idx] = view.original_image = torch.from_numpy(gt_jpg).float().permute(2,0,1).cuda()
            # Save intrinsic and extrinsic
            K = view.get_k().detach().cpu().numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            world_to_cam_list = view.world_view_transform.T[:3].detach().cpu().numpy()
            intrinsic_list.append([fx, fy, cx, cy])
            axis_angle = pytorch3d.transforms.matrix_to_axis_angle(torch.from_numpy(world_to_cam_list[:3, :3]).float())
            translation = torch.from_numpy(world_to_cam_list[:3, 3]).float()
            rot_trans = torch.concat((axis_angle, translation), dim=0)
            extrinsic_list.append(rot_trans)
        intrinsic_list = np.stack(intrinsic_list)
        extrinsic_list = np.stack(extrinsic_list)
        np.save(f"{misc_path}/test_intrinsic.npy", intrinsic_list)
        np.save(f"{misc_path}/test_extrinsic.npy", extrinsic_list)
        # Read the final total size (= stored misc data + Gaussians + color_aggregation_network)
        total_size = get_folder_file_size(misc_path) \
                    + get_folder_file_size(f"{model_path}/point_cloud/iteration_{iteration}/point_cloud.ply")
        if opt.use_color_aggregation:
            total_size += get_folder_file_size(
                f"{model_path}/color_aggregate_checkpoint/{iteration}/color_aggregation_network.pth"
            )
        with open(os.path.join(model_path, "result_fps_mem.json"), 'w') as f: 
            json.dump(
                {"FPS": FPS, "memory": total_size, "num_gaussians": num_gaussians}
                , f)
        

    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image
        out = render(view, gaussians, scene, pipeline, args, background, 
                    learnt_normal=learnt_normal, nb_src_frames=nb_src_frames, buffer_length=buffer_length, depth_error_threshold=depth_error_threshold,
                    app_model=app_model, do_find_closest_frame=True, do_render_src_depth=True, render_geo=True, return_depth_normal=False)
        aggregate_image = predicted_residual = None
        if opt.use_color_aggregation and color_aggregation_network is not None:
            autocast_ctx = autocast() if opt.enable_mix_precision else nullcontext()
            with autocast_ctx:
                fusion_out_dict = fuse_color(
                    out,
                    color_aggregation_network=color_aggregation_network,
                    iter_count=None,
                    burn_start=None,
                    burn_end=None,
                    iteration=iteration,
                    opts=opt,
                )
            if fusion_out_dict is not None:
                aggregate_image = fusion_out_dict["image_pred"]
                predicted_residual = fusion_out_dict["residual"]
                predicted_residual_vis = torch.abs(predicted_residual)
                predicted_residual_vis = predicted_residual_vis / torch.max(predicted_residual_vis)
            
        rendering = out["render"]
        _, H, W = rendering.shape

        depth = out[DEFAULT_DEPTH_TYPE].squeeze()
        depth_tsdf = depth.clone()

        depth = depth.detach().cpu().numpy()
        depth_color = colorize(out[DEFAULT_DEPTH_TYPE].detach().cpu().numpy(), cmap='magma_r')[:,:,:3]


        normal = out["rendered_normal"].permute(1,2,0)
        normal = (normal.detach().cpu()+1)/2
        normal = normal.detach().cpu().numpy()
        normal = (normal * 255).clip(0, 255).astype(np.uint8)
        if name == 'test':
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))

            if opt.use_color_aggregation and aggregate_image is not None and predicted_residual is not None:
                torchvision.utils.save_image(aggregate_image, os.path.join(render_aggregate_path, view.image_name + ".png"))
                torchvision.utils.save_image(predicted_residual_vis, os.path.join(residual_path, view.image_name + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        plt.imsave(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        plt.imsave(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out[f"{DEFAULT_DEPTH_TYPE}_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if bounds is not None:
                pts = gaussians.get_points_from_depth(view, ref_depth)
                unvalid_mask = (pts[...,0] < bounds[0,0]) | (pts[...,0] > bounds[0,1]) |\
                                (pts[...,1] < bounds[1,0]) | (pts[...,1] > bounds[1,1]) |\
                                (pts[...,2] < bounds[2,0]) | (pts[...,2] > bounds[2,1])
                unvalid_mask = unvalid_mask.reshape(H,W)
                ref_depth[unvalid_mask] = 0

            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2).detach().cpu().numpy()
            pose[:3, 3] = view.T.detach().cpu().numpy()
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            ref_depth = ref_depth.detach().cpu().numpy()
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)

def rendering(dataset, opt, pipeline, iteration, 
              skip_train, skip_test, max_depth, voxel_size, num_cluster, use_depth_filter, args):
    # print(f"multi_view_num {model.multi_view_num}")

    render_geo = args.render_geo
    learnt_normal = opt.learnt_normal
    nb_src_frames = opt.number_src_frames
    buffer_length = opt.buffer_length 

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, args, load_iteration=iteration, shuffle=False)

        # Load color aggregation network
        color_aggregation_network = None
        if opt.use_color_aggregation:
            color_aggregation_network = ColorFusionResidualNet(
                height=int(scene.getTrainCameras()[0].image_height * opt.residual_resolution_scale),
                width=int(scene.getTrainCameras()[0].image_width * opt.residual_resolution_scale),
                feat_aggregate_mode=opt.feat_aggregate_mode,
            ).cuda()
            color_aggregation_network.load_state_dict(torch.load(f"{args.model_path}/color_aggregate_checkpoint/{args.iteration}/color_aggregation_network.pth"))
            color_aggregation_network.eval()
            color_aggregation_network = torch.compile(color_aggregation_network, mode="max-autotune")

        bounds = None
        js_file = f"{scene.source_path}/transforms.json"
        if os.path.exists(js_file):
            with open(js_file) as file:
                meta = json.load(file)
                if "aabb_range" in meta:
                    bounds = (np.array(meta["aabb_range"]))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if bounds is not None:
            max_dis = np.max(bounds[:,1]-bounds[:,0])
            voxel_size = max_dis / 2048.0

        if render_geo:
            volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        else:
            volume = None

        # Get depth and rendered images
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                scene,
                gaussians,
                pipeline,
                args,
                opt,
                background,
                color_aggregation_network,
                learnt_normal=learnt_normal,
                nb_src_frames=nb_src_frames,
                buffer_length=buffer_length,
                max_depth=max_depth,
                volume=volume,
                use_depth_filter=use_depth_filter,
            )

            if render_geo:
                print(f"extract_triangle_mesh")
                mesh = volume.extract_triangle_mesh()
                path = os.path.join(dataset.model_path, "mesh")
                os.makedirs(path, exist_ok=True)
                
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                            write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

                mesh = post_process_mesh(mesh, num_cluster)
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                            write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                scene,
                gaussians,
                pipeline,
                args,
                opt,
                background,
                color_aggregation_network,
                learnt_normal=learnt_normal,
                nb_src_frames=nb_src_frames,
                buffer_length=buffer_length,
            )

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=20.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    # -------------- My added paramaters -----------------
    parser.add_argument('--render_geo', action='store_true')
    parser.add_argument('--src_image_ext', type=str, default="jpg")


    args = get_combined_args(parser)
    if args.iteration == -1: args.iteration = args.iterations

    if "kitchen" in args.model_path: setup_seed(678)
    else: setup_seed(22)
    
    rendering(model.extract(args), op.extract(args), pipeline.extract(args), args.iteration, 
            args.skip_train, args.skip_test, args.max_depth, 
            args.voxel_size, args.num_cluster, args.use_depth_filter, args)
    # All done
    print("\nRendering complete.")
