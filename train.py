
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

import os
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_photometric_ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import colorize
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
from torch.utils.tensorboard import SummaryWriter
from color_aggregation_network import ColorFusionResidualNet, fuse_color
import json

torch.backends.cudnn.benchmark = True

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model, learnt_normal, nb_src_frames, buffer_length,
                        depth_error_threshold, color_aggregation_network, opt):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = l1_aggregate_test = 0.0
                psnr_test = psnr_aggregate_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_result = renderFunc(viewpoint, scene.gaussians, scene, *renderArgs, 
                                                learnt_normal=learnt_normal, nb_src_frames=nb_src_frames, buffer_length=buffer_length, depth_error_threshold=depth_error_threshold,
                                                app_model=app_model, do_find_closest_frame=(config['name']=="test"))
                    # Get image
                    if render_result['app_image'] is None: rendered_image = render_result["render"]
                    else: rendered_image = render_result['app_image']
                    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
                    # Get depth
                    rendered_median_intersected_depth = render_result["median_intersected_depth"]
                    rendered_median_intersected_depth_cm = torch.from_numpy(colorize(rendered_median_intersected_depth.detach().cpu().numpy(), cmap='magma_r')[:,:,:3]).permute(2,0,1)
                    # Get normal
                    rendered_normal = render_result["rendered_normal"] # [3, 545, 977] # range from -1 to 1
                    rendered_normal_vis = (rendered_normal.detach().cpu()+1)/2
                    # Get gt image
                    gt_image = viewpoint.original_image.cuda()
                    # Color aggregation stuff
                    aggregated_image = None
                    residual_vis = None
                    if opt.use_color_aggregation and iteration >= opt.start_color_aggregation_iter:
                        fusion_out_dict = fuse_color(render_result, color_aggregation_network=color_aggregation_network, 
                                                    iter_count=None, burn_start=None, burn_end=None, 
                                                    iteration=iteration, opts=opt)
                        if fusion_out_dict is not None:
                            aggregated_image = fusion_out_dict["image_pred"]
                            residual_map = fusion_out_dict["residual"]
                            residual_vis = torch.abs(residual_map)
                            residual_vis = residual_vis / (residual_vis.max() + 1e-8)
                    else: fusion_out_dict = None

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/image_render".format(viewpoint.image_name), rendered_image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth_render_intersected_median".format(viewpoint.image_name), rendered_median_intersected_depth_cm[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/normal_render".format(viewpoint.image_name), rendered_normal_vis[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/image_gt".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if aggregated_image is not None:
                            tb_writer.add_images(config['name'] + "_view_{}/image_final".format(viewpoint.image_name), aggregated_image[None], global_step=iteration)
                        if residual_vis is not None:
                            tb_writer.add_images(config['name'] + "_view_{}/residual_vis".format(viewpoint.image_name), residual_vis[None], global_step=iteration)

                    l1_test += l1_loss(rendered_image, gt_image).mean().double()
                    psnr_test += psnr(rendered_image, gt_image).mean().double()
                    if fusion_out_dict is not None and aggregated_image is not None:
                        l1_aggregate_test += l1_loss(aggregated_image, gt_image).mean().double()
                        psnr_aggregate_test += psnr(aggregated_image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_aggregate_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                l1_aggregate_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if opt.use_color_aggregation and iteration >= opt.start_color_aggregation_iter:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_aggregate', l1_aggregate_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_aggregate', psnr_aggregate_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, args)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, photometric_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)


    learnt_normal = opt.learnt_normal
    nb_src_frames = opt.number_src_frames
    buffer_length = opt.buffer_length
    depth_error_threshold = opt.depth_error_threshold
    photo_ssim_weight = opt.photo_ssim_weight
    photo_weight = opt.photo_weight
    if learnt_normal: opt.scale_loss_weight = 0

    # Model loading
    checkpoint = args.start_checkpoint
    if checkpoint:
        (model_params, first_iter) = torch.load(f"{checkpoint}")
        gaussians.restore(model_params, opt)
        app_model.load_weights(checkpoint)

    # Create color aggregation network
    color_aggregation_network = None
    color_aggregation_optimizer = None
    if opt.use_color_aggregation:
        color_aggregation_network = ColorFusionResidualNet(
            height=int(scene.getTrainCameras()[0].image_height * opt.residual_resolution_scale),
            width=int(scene.getTrainCameras()[0].image_width * opt.residual_resolution_scale),
            feat_aggregate_mode=opt.feat_aggregate_mode,
        ).cuda()
        color_aggregation_optimizer = torch.optim.Adam(color_aggregation_network.parameters(), lr=0.001)
        os.makedirs(f"{args.model_path}/color_aggregate_checkpoint", exist_ok=True)

    # Define training parameters.
    epsilon = 1e-6 
    color_aggregate_iter_count = first_iter
    color_aggregate_burn_start = first_iter
    color_aggregate_burn_end = first_iter + opt.color_aggregate_burnin_steps


    # Store the rendered image and depth if resume training
    if first_iter > opt.single_view_weight_from_iter-len(scene.getTrainCameras())*2:
        iteration = first_iter
        with torch.no_grad():
            for viewpoint_cam in scene.getTrainCameras():
                if iteration > 1000 and opt.exposure_compensation: gaussians.use_app = True
                if (iteration - 1) == debug_from:  pipe.debug = True
                bg = torch.rand((3), device="cuda") if opt.random_background else background
                render_pkg = render(viewpoint_cam, gaussians, scene, pipe, args, bg, 
                                    learnt_normal=learnt_normal, nb_src_frames=nb_src_frames, buffer_length=buffer_length, depth_error_threshold=depth_error_threshold,
                                    app_model=app_model, render_geo=iteration>opt.single_view_weight_from_iter-len(scene.getTrainCameras())*2, 
                                    return_depth_normal=iteration>opt.single_view_weight_from_iter-len(scene.getTrainCameras())*2)
                image, viewspace_point_tensor, visibility_filter, radii = \
                    render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                if "median_intersected_depth" in render_pkg:
                    scene.rendered_depth_list[scene.getTrainCameras().index(viewpoint_cam)] = render_pkg["median_intersected_depth"].detach().to(device=args.data_device)
                
    # Main training loop
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    for iteration in range(first_iter, opt.iterations + 1):
        if iteration == opt.single_view_weight_from_iter:
            with torch.no_grad():
                gaussians._normal.copy_(gaussians.get_smallest_axis().data)

        if opt.use_color_aggregation and iteration in opt.color_aggregation_reduce_lr_iter:
            color_aggregation_optimizer.param_groups[0]['lr'] *= 0.5
            print("Reduce learning rate to:", color_aggregation_optimizer.param_groups[0]['lr'])

        iter_start.record()
        gaussians.update_learning_rate_offset(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam_idx = np.random.randint(0, len(viewpoint_stack))
        viewpoint_cam = viewpoint_stack.pop(cam_idx)

        gt_image = viewpoint_cam.original_image.cuda()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, scene, pipe, args, bg,
                            learnt_normal=learnt_normal, nb_src_frames=nb_src_frames, buffer_length=buffer_length, depth_error_threshold=depth_error_threshold,
                            app_model=app_model, render_geo=iteration>opt.single_view_weight_from_iter-len(scene.getTrainCameras())*2, 
                            return_depth_normal=iteration>opt.single_view_weight_from_iter-len(scene.getTrainCameras())*2)
        
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Cache the rendered depth
        if iteration>opt.single_view_weight_from_iter-len(scene.getTrainCameras())*2:
            scene.rendered_depth_list[scene.getTrainCameras().index(viewpoint_cam)] = render_pkg["median_intersected_depth"].detach().to(device=args.data_device)

        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if render_pkg['app_image'] is not None and ssim_loss < 0.5: Ll1 = l1_loss(render_pkg['app_image'], gt_image)
        else: Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        # single-view loss
        normal_loss = torch.tensor(0.0).float().cuda()
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["median_intersected_depth_normal"]
            normal_loss1 = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            normal_loss2 = weight * (1- ((depth_normal * normal).sum(0))).mean()
            normal_loss = (0.4*normal_loss1 + 0.6*normal_loss2)

        # multi-view loss
        photometric_loss = torch.tensor(0.0).float().cuda()
        if iteration > opt.multi_view_weight_from_iter:
            warped_image = render_pkg['warped_image'].view(-1,3,viewpoint_cam.image_height,viewpoint_cam.image_width)
            warped_image = warped_image[:opt.nb_visible_src_frames]
            cam_feat = render_pkg['cam_feat'].view(-1,4,viewpoint_cam.image_height,viewpoint_cam.image_width)[:opt.nb_visible_src_frames]

            valid_mask = torch.sum(cam_feat, dim=1, keepdim=True) > 0 # nb_visible_src_frames 1 H W
            ref_image = gt_image.unsqueeze(0) # 1 3 H W
            
            masked_warped_image = valid_mask.float()*warped_image + (1-valid_mask.float())*ref_image

            if torch.sum(valid_mask)>0:
                photometric_ssim_loss = 1 - torch.stack([compute_photometric_ssim(ref_image[0], masked_warped_image[i], size_average=False).mean(0) for i in range(len(masked_warped_image))])
                photometric_ssim_loss = torch.sum(photometric_ssim_loss*valid_mask[:,0]) / torch.sum(valid_mask[:,0])
                photometric_l1_loss =  torch.abs(ref_image-masked_warped_image).mean(1)
                photometric_l1_loss = torch.sum(photometric_l1_loss*valid_mask[:,0]) / torch.sum(valid_mask[:,0])
                photometric_loss = ((1-photo_ssim_weight)*photometric_l1_loss + photo_ssim_weight*photometric_ssim_loss)
                photometric_loss = photometric_loss * photo_weight
                photometric_loss = photometric_loss.mean()

            else: photometric_loss = torch.tensor(0.0).float().cuda()
        
        # ---------------------------------------------------
        # ------ Color aggragation network prediction -------
        # ---------------------------------------------------
        aggregate_image_loss = torch.tensor(0.0).float().cuda()
        aggregate_reg_loss = torch.tensor(0.0).float().cuda()
        burned_in_gauss = torch.tensor(0.0).float().cuda()
        fusion_out_dict = None
        if opt.use_color_aggregation and iteration > opt.start_color_aggregation_iter:
            # Prediction
            fusion_out_dict = fuse_color(render_pkg, color_aggregation_network=color_aggregation_network, 
                                        iter_count=color_aggregate_iter_count, burn_start=color_aggregate_burn_start, burn_end=color_aggregate_burn_end, 
                                        iteration=iteration ,opts=opt)
            if fusion_out_dict is not None: 
                image_pred = fusion_out_dict["image_pred"]
                burned_in_gauss = fusion_out_dict["burned_in_gauss"]
                aggregate_ssim_loss = (1.0 - ssim(image_pred, gt_image))
                aggregate_l1 = l1_loss(image_pred, gt_image)
                aggregate_image_loss = (1.0 - opt.lambda_dssim) * aggregate_l1 + opt.lambda_dssim * aggregate_ssim_loss
        # ---------------------------------------------------
        # ------ End aggregation network prediction ---------
        # ---------------------------------------------------
        
        loss = normal_loss + photometric_loss + aggregate_reg_loss
        if fusion_out_dict is None:
            loss += image_loss
        else:
            loss += (image_loss + aggregate_image_loss)/2
        if torch.isnan(loss): assert False, "Loss is NaN"
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * photometric_loss.item() if photometric_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, 
                            scene, render, (pipe, args, background), app_model, learnt_normal=learnt_normal, nb_src_frames=nb_src_frames, 
                            buffer_length=buffer_length, depth_error_threshold=depth_error_threshold, color_aggregation_network=color_aggregation_network, opt=opt)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            
            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                if (opt.opacity_decay > 0 and opt.opacity_decay < 1 and iteration % opt.opacity_decay_interval == 0 and iteration > opt.densify_from_iter):
                    gaussians.decay_opacity(opt.opacity_decay)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)
                # Color aggregation backward
                if opt.use_color_aggregation and iteration > opt.start_color_aggregation_iter:
                    color_aggregation_optimizer.step()
                    color_aggregation_optimizer.zero_grad()
                    color_aggregate_iter_count += 1

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)
                # Save the final color aggregation network
                if opt.use_color_aggregation and iteration > opt.start_color_aggregation_iter:
                    os.makedirs(f"{args.model_path}/color_aggregate_checkpoint/{iteration}", exist_ok=True)
                    torch.save(color_aggregation_network.state_dict(), f"{args.model_path}/color_aggregate_checkpoint/{iteration}/color_aggregation_network.pth")
                    torch.save(color_aggregation_optimizer.state_dict(), f"{args.model_path}/color_aggregate_checkpoint/{iteration}/color_aggregation_optimizer.pth")

    app_model.save_weights(scene.model_path, opt.iterations)
    # Save the final color aggregation network
    if opt.use_color_aggregation:
        os.makedirs(f"{args.model_path}/color_aggregate_checkpoint/{iteration}", exist_ok=True)
        torch.save(color_aggregation_network.state_dict(), f"{args.model_path}/color_aggregate_checkpoint/{iteration}/color_aggregation_network.pth")
        torch.save(color_aggregation_optimizer.state_dict(), f"{args.model_path}/color_aggregate_checkpoint/{iteration}/color_aggregation_optimizer.pth")




if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    setup_seed(24)

    args_dict = vars(args)
    # Write the dictionary to a JSON file:
    os.makedirs(args.model_path, exist_ok=True)
    with open(f'{args.model_path}/config.json', 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    print("Optimizing " + args.model_path)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    print("\nTraining complete.")
