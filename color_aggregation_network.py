import torch
import torch.nn.functional as F
from torch import nn


class ConvDecoderAE(nn.Module):
    """Two-level hourglass with input skip for spatial residual refinement."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up2_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up1_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d((hidden_dim // 2) * 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fuse_input = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(hidden_dim, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        bottleneck = self.enc3(p2)

        u2 = F.interpolate(bottleneck, size=e2.shape[-2:], mode="nearest")
        u2 = self.up2_conv(u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        u1 = self.up1_conv(u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        fused = self.fuse_input(torch.cat([d1, x], dim=1))
        return self.final(fused)


class ColorFusionResidualNet(nn.Module):
    """Aggregates per-view features into a single residual correction."""

    def __init__(
        self,
        *,
        height,
        width,
        feat_aggregate_mode="mean",
        per_view_feat_dim=32,
    ):
        super().__init__()
        if feat_aggregate_mode not in {"mean", "max"}:
            raise ValueError(f"Unsupported feat_aggregate_mode '{feat_aggregate_mode}'")

        self.height = height
        self.width = width
        self.feat_aggregate_mode = feat_aggregate_mode
        self.per_view_feat_dim = per_view_feat_dim

        self.per_view_mlp = nn.Sequential(
            nn.Linear(7, per_view_feat_dim),
            nn.ReLU(),
            nn.Linear(per_view_feat_dim, per_view_feat_dim),
            nn.ReLU(),
        )

        cnn_input_dim = per_view_feat_dim + 6  # aggregated features + ray dir + gaussian color

        self.conv_decoder = ConvDecoderAE(cnn_input_dim)

    def forward(
        self,
        x_views,
        ray_dir,
        c_3dgs,
        valid_depth_mask=None,
    ):
        """Return per-pixel residual in RGB space.

        Args:
            x_views: (B, M, 7) stacked per-view features.
            ray_dir: (B, 3) normalized ray directions per pixel.
            c_3dgs: (B, 3) rendered Gaussian color per pixel.
            valid_depth_mask: unused placeholder kept for call compatibility.
        """
        del valid_depth_mask  # Compatibility hook; depth masking handled upstream.

        B, M, _ = x_views.shape
        features = self.per_view_mlp(x_views.view(B * M, -1)).view(B, M, -1)

        if self.feat_aggregate_mode == "mean":
            aggregated = features.mean(dim=1)
        else:
            aggregated = features.max(dim=1).values

        feat_grid = aggregated.T.view(1, self.per_view_feat_dim, self.height, self.width)
        ray_grid = ray_dir.T.view(1, 3, self.height, self.width)
        color_grid = c_3dgs.T.view(1, 3, self.height, self.width)

        cnn_input = torch.cat([feat_grid, ray_grid, color_grid], dim=1)
        residual = self.conv_decoder(cnn_input)
        return residual.permute(2, 3, 0, 1).view(B, 3)


def compute_exposure_affine_matrix(I_s_warp, I_r, valid_mask):
    with torch.no_grad():
        _, H, W = I_r.shape
        device, dtype = I_r.device, I_r.dtype
        valid_mask_2d = valid_mask[0]
        I_r_valid = I_r[:, valid_mask_2d]
        I_s_valid = I_s_warp[:, valid_mask_2d]
        N = I_r_valid.shape[1]
        ones_valid = torch.ones((1, N), device=device, dtype=dtype)
        X = torch.cat([I_r_valid, ones_valid], dim=0).T
        Y = I_s_valid.T
        A = torch.linalg.lstsq(X, Y).solution
        affine_matrix = A.T

    ones_full = torch.ones((1, H, W), device=device, dtype=dtype)
    I_r_aug = torch.cat([I_r, ones_full], dim=0)
    transformed_image = torch.einsum("ij,jhw->ihw", affine_matrix, I_r_aug)
    return transformed_image, affine_matrix


def fuse_color(render_pkg, color_aggregation_network, iter_count, burn_start, burn_end, iteration, opts):
    if color_aggregation_network is None:
        return None

    if iter_count is None or burn_start is None or burn_end is None:
        burned_in_gauss = 1.0
    else:
        burned_in_gauss = max(0.0, min(1.0, (iter_count - burn_start) / (burn_end - burn_start)))
        burned_in_gauss = (burned_in_gauss + 1) / 2

    # Block the gradients to the Gaussians when the predicted residual are not fully used. 
    if burned_in_gauss < 1.0:
        rendered_image = render_pkg["render"].detach()
        _, H, W = rendered_image.shape
        feat = render_pkg["cam_feat"].detach()
        warped_image_list = render_pkg["warped_image"].view(-1, 3, H, W).permute(2, 3, 0, 1).detach()
        min_depth_diff = render_pkg["min_depth_diff"].detach()
        camera_ray_world = render_pkg["camera_ray"].view(3, H, W).detach()
    else:
        rendered_image = render_pkg["render"]
        _, H, W = rendered_image.shape
        feat = render_pkg["cam_feat"]
        warped_image_list = render_pkg["warped_image"].view(-1, 3, H, W).permute(2, 3, 0, 1)
        min_depth_diff = render_pkg["min_depth_diff"]
        camera_ray_world = render_pkg["camera_ray"].view(3, H, W)

    if opts.enable_exposure_correction:
        use_first_src_mask = render_pkg["use_first_src_frame_mask"]
        first_warped_image = warped_image_list[:, :, 0, :].permute(2, 0, 1) * use_first_src_mask
        rendered_image, _ = compute_exposure_affine_matrix(first_warped_image, rendered_image, use_first_src_mask == 1)

    feat = feat.view(-1, 4, H, W).permute(2, 3, 0, 1)
    nb_valid_warp_level = torch.sum(warped_image_list, dim=(0, 1, 3))
    nb_valid_warp_level = min(torch.count_nonzero(nb_valid_warp_level).item(), opts.nb_visible_src_frames)
    if nb_valid_warp_level == 0:
        return None

    feat = feat[:, :, :nb_valid_warp_level]
    warped_image_list = warped_image_list[:, :, :nb_valid_warp_level]
    valid_depth_list = (torch.sum(feat, dim=-1, keepdim=True) > 0.0).float()
    residual_list = (warped_image_list - rendered_image.permute(1, 2, 0).unsqueeze(2)) * valid_depth_list
    feat = torch.cat([residual_list, feat], dim=-1)
    valid_warp_mask = (min_depth_diff < 0.999).float()

    if opts.residual_resolution_scale != 1:
        feat_resize = F.interpolate(
            feat.permute(2, 3, 0, 1),
            scale_factor=opts.residual_resolution_scale,
            mode="bilinear",
            align_corners=True,
        ).permute(2, 3, 0, 1)
        rendered_image_resize = F.interpolate(
            rendered_image.unsqueeze(0),
            scale_factor=opts.residual_resolution_scale,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        camera_ray_world_resize = F.interpolate(
            camera_ray_world.unsqueeze(0),
            scale_factor=opts.residual_resolution_scale,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        camera_ray_world_resize = camera_ray_world_resize / (
            torch.norm(camera_ray_world_resize, dim=0, keepdim=True) + 1e-10
        )
        H_resize, W_resize = camera_ray_world_resize.shape[1], camera_ray_world_resize.shape[2]
    else:
        feat_resize = feat
        rendered_image_resize = rendered_image
        camera_ray_world_resize = camera_ray_world
        H_resize, W_resize = H, W

    feat_flat = feat_resize.view(-1, feat.shape[2], feat.shape[3]).contiguous()
    rendered_image_flat = rendered_image_resize.permute(1, 2, 0).view(-1, 3)
    camera_ray_flat = camera_ray_world_resize.view(3, -1).T

    residual = color_aggregation_network(feat_flat, camera_ray_flat, rendered_image_flat, None)
    residual = residual.T.view(3, H_resize, W_resize)

    if opts.residual_resolution_scale != 1:
        residual = torch.nn.functional.interpolate(
            residual.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=True
        ).squeeze(0)

    image_pred = burned_in_gauss * rendered_image + residual

    return {
        "image_pred": image_pred,
        "warped_image_list": warped_image_list,
        "residual": residual,
        "valid_warp_mask": valid_warp_mask,
        "burned_in_gauss": burned_in_gauss,
        "nb_valid_warp_level": nb_valid_warp_level,
    }
