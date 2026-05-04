"""
Loss functions for training the Grassmann model against video frames.

We provide:
  - L1 loss (standard, robust).
  - A lightweight "structural" loss based on local mean/variance matching,
    playing the role of SSIM without external dependencies. This gives the
    training signal beyond pixel-wise L1 (edges, blobs, gradients).
  - Optional LPIPS wrapper (requires `pip install lpips` and a pretrained
    network download on first use). Used when available; otherwise falls
    back to the structural loss.
  - Optional temporal LPIPS (equivalent to LPIPS on frame-differences),
    matching the video_lpips / temporal_lpips pattern in the Grassmann paper §3.4.

All losses take images shaped (H, W, 3) or (B, H, W, 3) and return a scalar.
Values are assumed to be in [0, 1].
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---- L1 ---------------------------------------------------------------------

def l1_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """Simple mean L1 distance. Both tensors same shape."""
    return (rendered - target).abs().mean()


def mse_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """Mean squared error. Both tensors same shape, values in [0, 1]."""
    return ((rendered - target) ** 2).mean()


def psnr(rendered: Tensor, target: Tensor) -> Tensor:
    """Peak signal-to-noise ratio in dB, assuming inputs in [0, 1] (max=1).

    PSNR = 10 * log10(MAX^2 / MSE). Returns a scalar tensor.
    """
    mse = mse_loss(rendered, target).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


# ---- Structural loss: a cheap SSIM substitute --------------------------------

def _to_bchw(img: Tensor) -> Tensor:
    """Convert (H, W, 3) or (B, H, W, 3) to (B, 3, H, W)."""
    if img.dim() == 3:
        img = img.unsqueeze(0)                      # (1, H, W, 3)
    return img.permute(0, 3, 1, 2)                  # (B, 3, H, W)


def _gaussian_kernel_1d(window: int, sigma: float, dtype: torch.dtype, device) -> Tensor:
    half = (window - 1) / 2.0
    x = torch.arange(window, dtype=dtype, device=device) - half
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def ssim_loss(rendered: Tensor, target: Tensor, window: int = 11, sigma: float = 1.5) -> Tensor:
    """1 - SSIM over the image, using the Gaussian-windowed SSIM as in 3DGS.

    Matches the standard 3DGS structural-similarity loss: 11x11 Gaussian window
    (sigma=1.5), C1=(0.01)^2, C2=(0.03)^2 on [0,1] images. Returns 1 - mean
    SSIM (DSSIM convention used in 3DGS as the structural-loss term).

    rendered, target: (H, W, 3) or (B, H, W, 3) in [0, 1].
    """
    r = _to_bchw(rendered)
    t = _to_bchw(target)
    dtype, device = r.dtype, r.device
    k1d = _gaussian_kernel_1d(window, sigma, dtype, device)
    kernel = (k1d[:, None] * k1d[None, :]).expand(3, 1, window, window).contiguous()
    pad = window // 2

    def conv(x: Tensor) -> Tensor:
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, groups=3)

    mu_r = conv(r)
    mu_t = conv(t)
    mu_r2 = mu_r * mu_r
    mu_t2 = mu_t * mu_t
    mu_rt = mu_r * mu_t
    sigma_r2 = conv(r * r) - mu_r2
    sigma_t2 = conv(t * t) - mu_t2
    sigma_rt = conv(r * t) - mu_rt
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2.0 * mu_rt + C1) * (2.0 * sigma_rt + C2)) / (
        (mu_r2 + mu_t2 + C1) * (sigma_r2 + sigma_t2 + C2)
    )
    return 1.0 - ssim_map.mean()


def structural_loss(rendered: Tensor, target: Tensor, window: int = 7) -> Tensor:
    """Local-mean + local-variance matching, averaged over channels.

    For each image we compute a box-filter local mean mu and variance sigma^2
    at every pixel (reflection padding). The loss is
        |mu_rendered - mu_target|  +  |sigma_rendered^2 - sigma_target^2|
    averaged over pixels and channels. This captures local brightness and
    texture mismatch without requiring LPIPS-style pretrained features.
    """
    r = _to_bchw(rendered)
    t = _to_bchw(target)
    pad = window // 2
    kernel = torch.ones(3, 1, window, window, dtype=r.dtype, device=r.device) / (window * window)

    def local_stats(x: Tensor) -> tuple[Tensor, Tensor]:
        # Use groups=3 so each channel gets its own spatial average.
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        mu = F.conv2d(x_pad, kernel, groups=3)
        mu2 = F.conv2d(x_pad * x_pad, kernel[:, :1].expand(3, 1, window, window), groups=3)
        sigma_sq = (mu2 - mu * mu).clamp_min(0.0)
        return mu, sigma_sq

    mu_r, var_r = local_stats(r)
    mu_t, var_t = local_stats(t)
    mean_err = (mu_r - mu_t).abs().mean()
    var_err = (var_r - var_t).abs().mean()
    return mean_err + var_err


# ---- Optional LPIPS ---------------------------------------------------------

class LPIPSLoss:
    """Lazy wrapper around the `lpips` package if available.

    Usage:
        loss_fn = LPIPSLoss(net='alex')  # may raise if package not installed
        val = loss_fn(rendered, target)

    We default to 'alex' (fast). Pass net='vgg' for slightly more perceptually
    accurate (and heavier) features.
    """

    def __init__(self, net: str = "alex", device: str = "cpu"):
        try:
            import lpips  # type: ignore
        except ImportError as e:
            raise ImportError(
                "LPIPS is optional. Install with `pip install lpips`."
            ) from e
        self._fn = lpips.LPIPS(net=net).to(device)
        self._fn.eval()
        for p in self._fn.parameters():
            p.requires_grad_(False)

    def __call__(self, rendered: Tensor, target: Tensor) -> Tensor:
        """Returns a scalar LPIPS distance in [0, 1]-ish.

        LPIPS expects inputs in [-1, 1] with shape (B, 3, H, W).
        """
        r = _to_bchw(rendered) * 2.0 - 1.0
        t = _to_bchw(target) * 2.0 - 1.0
        return self._fn(r, t).mean()


# ---- Temporal losses -------------------------------------------------------

def frame_differences(frames: Tensor) -> Tensor:
    """Given a video of shape (T, H, W, 3), return (T-1, H, W, 3) frame diffs."""
    return frames[1:] - frames[:-1]


def temporal_l1_loss(rendered_video: Tensor, target_video: Tensor) -> Tensor:
    """L1 on frame-differences. Encourages matching motion, not just pixels.

    rendered_video, target_video: (T, H, W, 3).
    """
    return l1_loss(frame_differences(rendered_video), frame_differences(target_video))


# ---- Combined multi-view loss ----------------------------------------------

def photometric_loss(
    rendered: Tensor,
    target: Tensor,
    *,
    lambda_l1: float = 0.8,
    lambda_structural: float = 0.2,
    structural_kind: str = "boxstats",
    lpips_fn: Optional[LPIPSLoss] = None,
    lambda_lpips: float = 0.0,
) -> Tensor:
    """Weighted sum of L1 + structural + optional LPIPS.

    rendered, target: (H, W, 3) or (B, H, W, 3) in [0, 1].
    structural_kind: 'boxstats' (legacy 7x7 local-mean+var) or 'ssim'
        (1 - SSIM, Gaussian-windowed, matches 3DGS).
    """
    loss = lambda_l1 * l1_loss(rendered, target)
    if lambda_structural > 0:
        if structural_kind == "ssim":
            loss = loss + lambda_structural * ssim_loss(rendered, target)
        elif structural_kind == "boxstats":
            loss = loss + lambda_structural * structural_loss(rendered, target)
        else:
            raise ValueError(f"unknown structural_kind: {structural_kind!r}")
    if lpips_fn is not None and lambda_lpips > 0:
        loss = loss + lambda_lpips * lpips_fn(rendered, target)
    return loss


# ---- 2DGS regularizers (depth distortion + normal consistency) -------------
#
# These are gated by --use_2dgs_losses and only meaningful when rendering through
# diff_surfel_rasterization (the channels rend_dist / rend_normal come from
# allmap, see grassmann.surfel_rasterizer.RENDER_PKG_KEYS).
#
# Reference: Huang et al., "2D Gaussian Splatting for Geometrically Accurate
# Radiance Fields", SIGGRAPH 2024, Eqs. 13-15.

def depth_distortion_loss(rend_dist: Tensor) -> Tensor:
    """2DGS Eq. 13: mean of the per-pixel ray-splat distortion map."""
    return rend_dist.mean()


def normal_consistency_loss(
    rend_normal: Tensor,
    surf_normal: Tensor,
) -> Tensor:
    """2DGS Eq. 14-15: 1 - <rend_normal, surf_normal> averaged over valid pixels.

    Both normals are (3, H, W), unit-norm, in the same coordinate frame.
    surf_normal is typically alpha-masked outside the silhouette.
    """
    dot = (rend_normal * surf_normal).sum(dim=0)            # (H, W)
    return (1.0 - dot).mean()


def depth_to_world_normal(
    depth: Tensor,
    cam,
) -> Tensor:
    """Back-project a depth map to world points and finite-difference for normals.

    Adapted from 2DGS utils/point_utils.py:depth_to_normal but using our Camera
    convention (`Camera.R`: world->camera, `Camera.c`: world-space cam center).

    depth: (1, H, W) — surface-depth map (median or expected).
    cam:   grassmann.projection.Camera

    Returns: (H, W, 3) world-space surface normal (unit), zeroed at the boundary.
    """
    H, W = depth.shape[-2:]
    device, dtype = depth.device, depth.dtype

    # Build a (H, W, 3) per-pixel ray direction in world frame.
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    # Camera-frame ray dir at depth=1: ((u-cx)/fx, (v-cy)/fy, 1).
    cam_x = (grid_x - cam.cx) / cam.fx
    cam_y = (grid_y - cam.cy) / cam.fy
    cam_z = torch.ones_like(cam_x)
    ray_cam = torch.stack([cam_x, cam_y, cam_z], dim=-1)    # (H, W, 3)

    # cam->world: world = c + R^T @ cam_point.
    R = cam.R.to(device=device, dtype=dtype)
    c = cam.c.to(device=device, dtype=dtype)
    ray_world = ray_cam @ R                                  # (H, W, 3); R^T @ v == v @ R

    points = depth.permute(1, 2, 0) * ray_world + c          # (H, W, 3)

    # Finite-difference cross product (matches 2DGS).
    output = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]                # (H-2, W-2, 3) vertical
    dy = points[1:-1, 2:] - points[1:-1, :-2]                # (H-2, W-2, 3) horizontal
    n = torch.linalg.cross(dx, dy, dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)
    output[1:-1, 1:-1, :] = n
    return output                                            # (H, W, 3) world-frame
