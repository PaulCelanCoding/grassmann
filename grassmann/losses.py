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
    lpips_fn: Optional[LPIPSLoss] = None,
    lambda_lpips: float = 0.0,
) -> Tensor:
    """Weighted sum of L1 + structural + optional LPIPS.

    rendered, target: (H, W, 3) or (B, H, W, 3) in [0, 1].
    """
    loss = lambda_l1 * l1_loss(rendered, target)
    if lambda_structural > 0:
        loss = loss + lambda_structural * structural_loss(rendered, target)
    if lpips_fn is not None and lambda_lpips > 0:
        loss = loss + lambda_lpips * lpips_fn(rendered, target)
    return loss
