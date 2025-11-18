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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import torch
import torch.nn.functional as F

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, args_all, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.source_path = args.source_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f"cameras_extent {self.cameras_extent}")

        # Nearby camera loading (following PGSR: https://github.com/zju3dv/PGSR)
        self.multi_view_num = args.multi_view_num
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            train_cameras = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.train_cameras[resolution_scale] = train_cameras

            print("Loading Test Cameras")
            test_cameras = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            self.test_cameras[resolution_scale] = test_cameras

            self._initialize_train_buffers(train_cameras, args)
            train_metrics = self._compute_train_metrics()
            self._write_train_multiview(train_cameras, train_metrics, args, args_all)

            if args.eval and test_cameras:
                test_metrics = self._compute_test_metrics(test_cameras)
                self._write_test_multiview(test_cameras, train_cameras, test_metrics, args, args_all)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, mask=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mask)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def _initialize_train_buffers(self, train_cameras, args):
        print("computing nearest_id")
        world_view_transforms = []
        original_images = []
        rendered_depths = []
        camera_centers = []
        center_rays = []

        for cam in train_cameras:
            world_view_transforms.append(cam.world_view_transform.T)
            original_images.append(cam.original_image)
            rendered_depths.append(
                torch.zeros(1, cam.image_height, cam.image_width, device=args.data_device)
            )
            camera_centers.append(cam.camera_center)

            rotation = torch.tensor(cam.R).float().cuda()
            forward_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
            center_rays.append(forward_ray @ rotation.transpose(-1, -2))

        self.world_view_transforms = torch.stack(world_view_transforms)
        self.original_image_list = torch.stack(original_images)
        self.rendered_depth_list = torch.stack(rendered_depths)

        for idx, cam in enumerate(train_cameras):
            cam.original_image = self.original_image_list[idx]

        self.camera_centers = torch.stack(camera_centers, dim=0)
        self.center_rays = F.normalize(torch.stack(center_rays, dim=0), dim=-1)

    def _compute_train_metrics(self):
        diss = torch.norm(
            self.camera_centers[:, None] - self.camera_centers[None], dim=-1
        ).detach().cpu().numpy()

        dot_products = torch.sum(self.center_rays[:, None] * self.center_rays[None], dim=-1)
        angles = torch.arccos(dot_products) * 180 / 3.14159
        angles = angles.detach().cpu().numpy()

        relative_poses = torch.matmul(
            self.world_view_transforms.unsqueeze(0),
            torch.inverse(self.world_view_transforms).unsqueeze(1),
        )
        cam_diff = torch.mean(
            torch.abs(
                relative_poses
                - torch.eye(4).float().cuda().unsqueeze(0).unsqueeze(0)
            ),
            dim=[2, 3],
        ).detach().cpu().numpy()

        return diss, angles, cam_diff

    def _write_train_multiview(self, train_cameras, metrics, args, args_all):
        diss, angles, cam_diff = metrics
        json_path = os.path.join(self.model_path, "multi_view.json")
        with open(json_path, "w") as file:
            for cam_id, cam in enumerate(train_cameras):
                candidate_indices = self._filtered_indices(diss[cam_id], angles[cam_id], args)
                ordered_indices = self._ordered_neighbors(candidate_indices, cam_diff[cam_id], args_all)
                self._record_neighbors(cam, ordered_indices, train_cameras, file)

    def _compute_test_metrics(self, test_cameras):
        test_world_view_transforms = []
        test_camera_centers = []
        test_center_rays = []

        for cam in test_cameras:
            test_world_view_transforms.append(cam.world_view_transform.T)
            test_camera_centers.append(cam.camera_center)

            rotation = torch.tensor(cam.R).float().cuda()
            forward_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
            test_center_rays.append(forward_ray @ rotation.transpose(-1, -2))

        test_world_view_transforms = torch.stack(test_world_view_transforms)
        test_camera_centers = torch.stack(test_camera_centers, dim=0)
        test_center_rays = F.normalize(torch.stack(test_center_rays, dim=0), dim=-1)

        diss_test = torch.norm(
            test_camera_centers[:, None] - self.camera_centers[None], dim=-1
        ).detach().cpu().numpy()

        dot_products = torch.sum(test_center_rays[:, None] * self.center_rays[None], dim=-1)
        angles_test = torch.arccos(dot_products) * 180 / 3.14159
        angles_test = angles_test.detach().cpu().numpy()

        test_relative_poses = torch.matmul(
            test_world_view_transforms.unsqueeze(0),
            torch.inverse(self.world_view_transforms).unsqueeze(1),
        )
        test_cam_diff = torch.mean(
            torch.abs(
                test_relative_poses
                - torch.eye(4).float().cuda().unsqueeze(0).unsqueeze(0)
            ),
            dim=[2, 3],
        ).detach().cpu().numpy()

        return diss_test, angles_test, test_cam_diff

    def _write_test_multiview(self, test_cameras, train_cameras, metrics, args, args_all):
        diss_test, angles_test, test_cam_diff = metrics
        json_path = os.path.join(self.model_path, "multi_view_test.json")
        with open(json_path, "w") as file:
            for cam_id, cam in enumerate(test_cameras):
                candidate_indices = self._filtered_indices(diss_test[cam_id], angles_test[cam_id], args)
                ordered_indices = self._ordered_neighbors(
                    candidate_indices, test_cam_diff[:, cam_id], args_all
                )
                self._record_neighbors(cam, ordered_indices, train_cameras, file)

    def _filtered_indices(self, distance_row, angle_row, args):
        sorted_indices = np.lexsort((angle_row, distance_row))
        mask = (
            (angle_row[sorted_indices] < args.multi_view_max_angle)
            & (distance_row[sorted_indices] > args.multi_view_min_dis)
            & (distance_row[sorted_indices] < args.multi_view_max_dis)
        )
        return sorted_indices[mask]

    def _ordered_neighbors(self, sorted_indices, cam_diff_values, args_all):
        if len(sorted_indices) == 0:
            return sorted_indices

        multi_view_num = min(self.multi_view_num, len(sorted_indices))
        selected = sorted_indices[:multi_view_num]
        cam_diff = cam_diff_values[selected]
        smallest_index = selected[np.argmin(cam_diff)]

        if args_all.enable_exposure_correction and len(selected) > 1:
            selected_list = selected.tolist()
            selected_list.remove(smallest_index)
            selected_list.insert(0, smallest_index)
            selected = np.array(selected_list)

        return selected

    def _record_neighbors(self, camera, neighbor_indices, reference_cameras, file_handle):
        camera.nearest_id = []
        camera.nearest_names = []

        json_d = {"ref_name": camera.image_name, "nearest_name": []}
        for index in neighbor_indices:
            camera.nearest_id.append(int(index))
            nearest_name = reference_cameras[index].image_name
            camera.nearest_names.append(nearest_name)
            json_d["nearest_name"].append(nearest_name)

        json_str = json.dumps(json_d, separators=(",", ":"))
        file_handle.write(json_str + "\n")
