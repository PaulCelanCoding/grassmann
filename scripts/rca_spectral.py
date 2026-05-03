"""
Spectral RCA on a 3-plane (G(3,4)) Grassmann GS checkpoint.

Reads a checkpoint produced by train_mono.py, reconstructs the per-Gaussian
parameters, and prints distributions of:

  * Sigma_3D(t_0) eigenvalues (the rank-2 disk shape; one of three is
    numerical zero, the other two are the disk's axes).
  * disk anisotropy = lambda_max / lambda_min (over the two non-zero eigs)
  * disk area = pi * sqrt(lambda_max * lambda_min)
  * Sigma_tt_pure (temporal extent in scene-time units)
  * |c_world| (spatial-temporal cross-coupling magnitude)
  * opacity (after sigmoid)
  * n_hat alignment with the time axis (n_t component)

Output: prints a markdown table to stdout. Each row is a metric with
percentiles 1, 25, 50, 75, 99.

Usage:
  python scripts/rca_spectral.py checkpoints/<run>/trained_nerfies_random.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Ensure repo on path.
import sys
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from grassmann.gaussian import GaussianParams, compute_derived, condition_on_time


def _percentiles(x: torch.Tensor, ps=(1, 25, 50, 75, 99)) -> list[float]:
    if x.numel() == 0:
        return [float("nan")] * len(ps)
    xs = x.detach().double().cpu().numpy()
    return [float(np.percentile(xs, p)) for p in ps]


def _row(name: str, x: torch.Tensor, fmt: str = "{:.4g}") -> str:
    p = _percentiles(x)
    return ("| " + name + " | " + " | ".join(fmt.format(v) for v in p) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--t0", type=float, default=0.5,
                    help="Time at which to evaluate Σ_3D(t_0). Defaults to 0.5 "
                         "(midpoint of normalized time range).")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Reconstruct GaussianParams from state_dict (apply the same forward-pass
    # reparameterizations as TrainableGaussians.forward).
    n_raw = state["n_raw"].double()
    L_raw = state["L_raw"].double()
    mu = state["mu"].double()
    opacity_logit = state["opacity_logit"].double()

    n_unit = n_raw / n_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    opacity = torch.sigmoid(opacity_logit)
    if "color_logit" in state:
        color = torch.sigmoid(state["color_logit"].double())
    elif "sh_dc" in state:
        from grassmann.gaussian import sh_dc_to_rgb
        color = sh_dc_to_rgb(state["sh_dc"].double())
    else:
        raise KeyError("Checkpoint has neither `color_logit` nor `sh_dc`.")

    params = GaussianParams(
        n=n_unit, L_raw=L_raw, mu=mu, opacity=opacity, color=color,
        sigma_k_pixel=1.0, sigma_k_temporal=0.0,
    )
    N = n_unit.shape[0]
    print(f"# Spectral RCA: {args.ckpt.name}")
    print()
    print(f"N = {N} Gaussians")
    print()

    # Σ_4D check: kernel contains n_hat?
    nL = torch.einsum("...i,...ij->...j", n_unit, L_raw)
    L_plane = L_raw - n_unit.unsqueeze(-1) * nL.unsqueeze(-2)
    Sigma_4D = L_plane @ L_plane.transpose(-1, -2)
    Sigma4D_n = (Sigma_4D @ n_unit.unsqueeze(-1)).squeeze(-1)
    null_norm = Sigma4D_n.norm(dim=-1)
    print(f"Σ_4D · n̂ residual (should be ~0): max = {null_norm.max():.2e}, "
          f"median = {null_norm.median():.2e}")
    print()

    # Time-conditioning at t_0
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t_0=args.t0)
    Sigma_3D_t = tc.Sigma_3D_t                                          # (N, 3, 3)
    eigs = torch.linalg.eigvalsh(Sigma_3D_t)                            # (N, 3) ascending
    lam_min_nonzero = eigs[:, 1]                                        # 2nd smallest (smallest non-zero)
    lam_max = eigs[:, 2]                                                # largest
    anisotropy = lam_max / lam_min_nonzero.clamp_min(1e-30)
    disk_area = np.pi * (lam_max * lam_min_nonzero).clamp_min(0).sqrt()
    sigma_tt_pure = derived._sigma_tt_pure                              # (N,)
    c_norm = derived.c_world.norm(dim=-1)                               # (N,)
    n_t_abs = n_unit[..., 0].abs()                                      # (N,)

    # Distributions table
    print("## Spectral distributions (percentiles 1 / 25 / 50 / 75 / 99)")
    print()
    print("| metric | p1 | p25 | p50 | p75 | p99 |")
    print("|---|---|---|---|---|---|")
    print(_row("Σ_3D(t_0) λ_min (disk minor axis var)", lam_min_nonzero))
    print(_row("Σ_3D(t_0) λ_max (disk major axis var)", lam_max))
    print(_row("anisotropy λ_max / λ_min", anisotropy))
    print(_row("disk area π·√(λ_max·λ_min)", disk_area))
    print(_row("Σ_tt_pure (temporal extent)", sigma_tt_pure))
    print(_row("|c_world| (space-time coupling)", c_norm))
    print(_row("opacity (after sigmoid)", opacity))
    print(_row("|n̂_t| (time-axis alignment)", n_t_abs))
    print()

    # Diagnostic counters
    near_zero_lam_min = (lam_min_nonzero < 1e-6).sum().item()
    huge_lam_max = (lam_max > 1.0).sum().item()
    high_aniso = (anisotropy > 100).sum().item()
    near_zero_sigma_tt = (sigma_tt_pure < 1e-6).sum().item()
    near_one_n_t = (n_t_abs > 0.95).sum().item()
    low_op = (opacity.squeeze() < 0.01).sum().item()

    print("## Pathology counts")
    print()
    print(f"- Collapsed disks (λ_min < 1e-6): **{near_zero_lam_min}** / {N}  "
          f"({100*near_zero_lam_min/N:.1f} %)")
    print(f"- Huge disks (λ_max > 1.0 in scene units²): **{huge_lam_max}** / {N}")
    print(f"- High anisotropy (λ_max/λ_min > 100): **{high_aniso}** / {N}  "
          f"({100*high_aniso/N:.1f} %)")
    print(f"- Near-degenerate temporal (Σ_tt < 1e-6): **{near_zero_sigma_tt}** / {N}")
    print(f"- n̂ near time-axis (|n_t| > 0.95): **{near_one_n_t}** / {N}  "
          f"({100*near_one_n_t/N:.1f} %)")
    print(f"- Effectively dead (opacity < 0.01): **{low_op}** / {N}  "
          f"({100*low_op/N:.1f} %)")


if __name__ == "__main__":
    main()
